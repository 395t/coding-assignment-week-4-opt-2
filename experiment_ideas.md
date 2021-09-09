# Experiment Ideas

## Experiment 1: CIFAR-10 Image Classification
* Number of classes: 10
* Network architectures considered:
  * Simple MLP
  * Simple CNN
  * Some medium-size ResNet variant


## Experiment 2: Tiny ImageNet Image Classification
* Number of classes: 200
* Network architectures considered:
  * Simple MLP
  * Simple CNN
  * Some medium-size ResNet variant


## Experiment 3: EURLEX57K text classification
* Task: Multi-label classification
* Number of labels: 4271 -- but maybe we should probably use the 746 "frequent" labels?
* Network architectures considered:
  * BERT: which one? (Maybe raw transformer?)
  * Basic RNN
  * Bi-Directional LSTM 


## Questions to answer
* What hyperparameters do we want to explore?
  * Learning rate
  * Network architecture (depth, type, activation, etc.)
  * Optimizer hyperparameters (e.g. momentum, weight decay)
  * Augmentation?
  * ?

* What ablation studies do we want to include?
  * Benefits of momentum
  * Benefits of per-parameter adaptation of learning rate
  * ?

# Goals
* implement the core idea of five papers on a standard architecture and task/dataset
* evaluate implementation on 3 different datasets
* compare the relative performance, stability (to hyper-parameters) across all datasets
* draw conclusions for which methods work best (or if there is no clear winner)
