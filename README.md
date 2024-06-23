# GPT-2 implementation in PyTorch

This repository contains a recreation of GPT-2 using PyTorch. It includes scripts for loading datasets, defining the model architecture, generating text, and interacting with the model using a Jupyter notebook. It is heavily based on the repo from Andrej Karpathy, which can be found [here](https://github.com/karpathy/build-nanogpt). Some of the code, like the dataset loading and tokenization, is taken directly from his repo.

### Installation

To get started, clone the repository and install the required packages:

```sh
git clone https://github.com/jorgensandhaug/gpt-2-from-scratch.git
cd gpt2_pytorch
pip install -r requirements.txt
```

### Usage

To train the model, you first need to download the fineweb dataset and tokenize it. This can be done by running the following command:

```sh
python fineweb.py
```

Then you can train the model by running:

```sh
python train.py
```