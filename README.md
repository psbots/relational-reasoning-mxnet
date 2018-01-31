## A Simple Neural Network Module for Relational Reasoning 
## Implemented in MXNet as part of [NIPS Implementation Challenge](https://nurture.ai/nips-challenge/)

Link to paper : [Arxiv](https://arxiv.org/abs/1706.01427) | [pdf](https://arxiv.org/pdf/1706.01427)

### About this implementation 
This implementation tries to replicate the results in the paper for the sort-of-clevr dataset. The code for generating the dataset has been borrowed from [here](https://github.com/kimhc6028/relational-networks/blob/master/sort_of_clevr_generator.py).

[**Apache MXNet**](https://mxnet.apache.org/) is used as the framework of choice, specifically MXNet's symbolic interface.

## Results

Baseline Model(CNN + MLP) Accuracy at 30th Epoch

| Overall       |0.639880|
|---------------|--------|
| Non-Relational|0.599121|
| Relational	|0.683105|

Model with Relation Network (CNN + RN) at 30th Epoch

| Overall       |0.899305|
|---------------|--------|
| Non-Relational|0.978027|
| Relational	|0.821289|