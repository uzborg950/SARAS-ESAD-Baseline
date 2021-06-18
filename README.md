# Surgeon Action Detection for endoscopic images/videos Using Deep Learning (MSc. Artificial Intelligence project)

### Introduction

The project is a fork of https://github.com/Viveksbawa/SARAS-ESAD-Baseline. It modifies and builds upon the existing RetinaNet architecture and improves its capability of recording temporal information. 

### Primary Features (so far)

- RetinaNet + LSTM to capture spatiotemporal features. 

### Fineprint
Due to computational limitations, all work is being demonstrated on the following specs:
- Resnet18
- Input size = 600
- Focal loss (Due to its capability of down-weighting easy examples)
- batch size = 16
