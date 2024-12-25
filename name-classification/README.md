# NLP From Scratch: Classifying Names with a Character-Level RNN

Re-implement from [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial)

## Environments

| | |
|:---:|:---:|
| GPU | NVIDIA GeForce GTX 1080 Ti |
| python | 3.8.18 |
| torch | torch 1.13.1+cu116 |

## Data

This repository uses [data from the original source](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial#preparing-the-data)

![Data format](./assets/dataformat.png)

## Data Processing

This repository classifies name with character-level, so with each character, convert to a **one-hot-vector**

![Onehot](./assets/onehot.png)

## Model

This repository uses basic RNN to classify

- RNN Cell

![RNNcell](./assets/RNNcell.png)

- RNN Pipeline (For example `Phong` as input)

![RNNpipeline](./assets/RNNpipeline.png)

## Training Results

- Loss of training progress (300K iter)

![loss](./assets/loss.png)

- Confusion matrix after training

![confusion](./assets/confusion.png)

## Predict

After training, you will have `rnn.pt`, using it to running inference with

``` bash
python predict.py -i Phong
# Name `Phong` is from `Vietnamese`
```
