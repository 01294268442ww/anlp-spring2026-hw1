#!/usr/bin/env bash

conda create -n llama_hw python=3.11.4
conda activate llama_hw

# Modify this command depending on your system's environment.
# As written, this command assumes you have CUDA on your machine, but
# refer to https://pytorch.org/get-started/previous-versions/ for the correct
# command for your system.
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install tqdm==4.66.1
pip install requests==2.32.5
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==1.2.2
pip install numpy==2.4.1
pip install tokenizers==0.13.3
pip install sentencepiece==0.1.99
pip install matplotlib==3.10.8
wget https://www.cs.cmu.edu/~vijayv/stories42M.pt