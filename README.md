
# Surgeon Action Detection for endoscopic images/videos Using Deep Learning (MSc. Artificial Intelligence project)

### Introduction

The project is a fork of https://github.com/Viveksbawa/SARAS-ESAD-Baseline. It modifies and builds upon the existing RetinaNet architecture and extends its capability to record spatiotemporal information. 

### Dataset
- 21 action classes
- YOLO style bounding box coordinate ground truths (center x, center y, width, height)
![hist_train_set_full-page-001](https://user-images.githubusercontent.com/16350367/124779586-579f2e80-df5b-11eb-8c3d-93525b243aa0.jpg)


### Primary Features (so far)
- Repeat last frames from individual surgical footage to create full mini-batches consisting of frames from the same footage (useful for stateful models).
- Temporal subnets (ConvLSTM, Conv2d, BatchNorm, Relu)
- Truncated backpropagation through time. Persisting states in forward passes. States reset before the next surgical footage is passed.
- Partial loading of model weights through config: --load_non_strict_pretrained and --pretrained_iter. For e.g. Train RetinaNet independently then use ResnetFPN weights in TempRetinaNet.
- Convert normal dataset to time slices by config: --time_distributed_backbone=True and --temporal_slice_timesteps. batch_size * temporal_slice_timesteps = Total batch size loaded in one step.
- Create high contrast image by: --shifted_mean=True
- Predict surgical phase in multi-task manner: --predict_surgical_phase=True and --num_phases

### Fineprint
Due to computational limitations, experiments are run in the following configs depending on what can be accomodated:
- Resnet18
- Input size = 200, 600
- Focal loss (Due to its capability of down-weighting easy examples)
- batch size = 2, 4, 8, 16

### Demo 
- Input size = 200
- Total batch size=16, where batch_size=4 and temporal_slice_timesteps=4
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
