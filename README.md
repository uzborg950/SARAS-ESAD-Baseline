
# Surgeon Action Detection for endoscopic images/videos Using Deep Learning (MSc. Artificial Intelligence project)

### Introduction

The project is a fork of https://github.com/Viveksbawa/SARAS-ESAD-Baseline. It modifies and builds upon the existing RetinaNet architecture and extends its capability to record spatiotemporal information. 

### Dataset
- 21 action classes
- YOLO style bounding box coordinate ground truths (center x, center y, width, height)
![hist_train_set_full-page-001](https://user-images.githubusercontent.com/16350367/124779586-579f2e80-df5b-11eb-8c3d-93525b243aa0.jpg)


### Primary Features (so far)

- RetinaNet + Temporal network of configurable depth n (ConvLSTM, Conv2d, BatchNorm, Relu) x n

### Fineprint
Due to computational limitations, all work is being demonstrated on the following specs:
- Resnet18
- Input size = 600
- Focal loss (Due to its capability of down-weighting easy examples)
- batch size = 16

### Demo 

- Temporal layer depth = 2
- Ground Truth = green

![ezgif com-gif-maker](https://user-images.githubusercontent.com/16350367/130421530-0c900896-a416-4a3c-af1c-8008cb5070e6.gif)



### Results
- Early stopping 
- epochs = 3.40 
- iterations = 4000

| IOU Threshold | mAP |
| ------------- | --- |
| 0.10          | 34.52 |
| 0.30          | 26.03 |
| 0.50          | 14.07 |
