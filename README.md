# Marlin-Mistral-7b-v0.1 FT Guide

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow)
![Datasets](https://img.shields.io/badge/Datasets-HuggingFace-green)
![Accelerate](https://img.shields.io/badge/Accelerate-1.x-lightblue)
![WandB](https://img.shields.io/badge/WandB-Logging-purple)
![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-blueviolet)



## Introduction

This repository contains a Jupyter notebook that serves as a follow along guide for my blog post surrounding this finetune.

## Instructions

This tutorial is around 80% modular for many models available on huggingface. If you prepare your own data, clean it, and fit it into this structure you should be able to train other models as well.

### Setting Up

1. Clone this repository to your local machine.
2. Create a virtual environment (optional but recommended).
3. Install the required libraries using the command `pip install -r requirements.txt`.

### Running the Jupyter Notebook

1. Navigate to the cloned repository.
2. Launch Jupyter Notebook by running `jupyter notebook`.
3. Open the `marlin-mistral-ft.ipynb` notebook.
4. Execute the cells in the notebook to run the code.

## Training Code (`training.py`)

The training script, `training.py`, covers the following steps:

1. Setting up the accelerator.
2. Loading and processing the training dataset.
3. Loading the base model and tokenizer.
4. Tokenizing and formatting the dataset.
5. Checking and evaluating the base model.
6. Fine-tuning the model using the Peft approach.
7. Training the model using the Transformers library.
8. Evaluating the fine-tuned model.
9. Pushing the model to the HuggingFace Hub.

To run the training script:

```bash
python training.py
```
The current setup for this script will evaluate the original model and the finetuned model based on the loaded dataset.

## Acknowledgements

[mistralai/mistral-7b-v0.1 Paper](https://doi.org/10.48550/arXiv.2310.06825)
