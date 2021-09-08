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
  * BERT: which one?


## Questions to answer
* What hyperparameters do we want to explore?
  * Learning rate
  * Network architecture (depth, type, activation, etc.)
  * Optimizer hyperparameters (e.g. momentum)
  * ?

* What ablation studies do we want to include?
  * Benefits of momentum
  * Benefits of per-parameter adaptation of learning rate
  * ?

