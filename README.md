# GALNet
Official implementation of our method for gaze estimation with geometric-aware representation learning.

## Usage

The pipeine consists of two main steps:
1. Prepare the data and perform 3DMM fitting.
2. Run training and evaluation.

For convenience, we provide a shell script to run MPIIFaceGaze leave-one-person out evaluation:

```
bash train_aaai.sh
```

This script will automatically train and test the model across subjects.

## Environments
We provide two separate environments:
1. Training
2. 3DMM fitting
Create them using the provided requirement files:

```
# Training environment
conda create -n gaze_train python=3.8
conda activate gaze_train
pip install -r requirements.txt

# 3DMM fitting environment
conda create -n gaze_flame python=3.9
conda activate gaze_flame
pip install -r requirements_flame.txt

```

## 3DMM Fitting
To perform single-image 3DMM fitting, run:

```
python photometric_fitting.py
```
To be noted, you should change the 'input_folder' to your data root.
