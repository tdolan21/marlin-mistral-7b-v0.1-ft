from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from huggingface_hub import login

login()


## Load accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Load the full training dataset
full_train_dataset = load_dataset('vicgalle/alpaca-gpt4', split='train')

# Shuffle and select the first 10,000 samples
subset_dataset = full_train_dataset.shuffle(seed=42).select(range(10000))

# Split the subset into train, validation, and test sets
train_dataset = subset_dataset.train_test_split(test_size=0.2)['train']  # 8,000 samples for training
temp_dataset = subset_dataset.train_test_split(test_size=0.2)['test']   # 2,000 samples left
eval_dataset = temp_dataset.train_test_split(test_size=0.5)['train']    # 1,000 samples for validation
test_dataset = temp_dataset.train_test_split(test_size=0.5)['test']     # 1,000 samples for testing

# Print the datasets
print(train_dataset)
print(eval_dataset)
print(test_dataset)

## Load Base Model
base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

## Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)

## setup tokenize function to make labels and input_ids the same.
tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

## Format all samples into this prompt format
def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}
"""
    return tokenize(full_prompt)

# tokenize each sample based on the prompt format
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print(tokenized_train_dataset[4]['input_ids'])

# check that the sample has the max length of 512
print(len(tokenized_train_dataset[4]['input_ids']))

## Check and evaluate the base model
print("Instruction: " + test_dataset[1]['instruction'])
print("Input: " + test_dataset[1]['input'])
print("Response: " + test_dataset[1]['output'] + "\n")

eval_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{test_dataset[1]['instruction']}

### Input:
{test_dataset[1]['input']}

### Response:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))


## Start finetuning with peft
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# print model to examine layers
print(model)


## Define the LoRa config
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)


## Print the model differences with the LoRa adapters
print(model)


## Track the training stats on wandb
import wandb, os
wandb.login()

wandb_project = "marlin-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project



## begin training
if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

import transformers
from datetime import datetime

project = "marlin-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token



from peft import PeftModel
base_model = model = AutoModelForCausalLM.from_pretrained("mistralai/mistral-7b-v0.1", trust_remote_code=True, torch_dtype=torch.float32)
ft_model = PeftModel.from_pretrained(base_model, "mistral-marlin-finetune/checkpoint-1000")

ft_model.eval()
with torch.no_grad():
        print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True))


## push to hub
model = ft_model.merge_and_load()
model.push_to_hub("macadeliccc/marlin-mistral-7b-v0.1")
