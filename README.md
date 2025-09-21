# Code for **Learning Geometric-aware Representation For Gaze Estimation**

## Usage
### Directly use our code

You can perform two steps to run our codes.
1. Prepare the data and fit the 3DMM model.
2. Run the commands.

We recommand you to run our code using

```
bash train_aaai.sh
```
This command include the training and testing process for MPIIFaceGaze leave-one-person-out evaluation.

## Environments
Our code include two environments for training and 3DMM fitting, respectively. You can use requirement.txt and requirement_flame.txt to create the enviroments.

## 3DMM Fitting
You can perform single-image 3DMM fitting using below command

```
python photometric_fitting.py
```
To be noted, you should change the 'input_folder' to your data root.
