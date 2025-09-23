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

## Citation
if you use this code, please consider citing our paper:

```
@INPROCEEDINGS{11084621,
  author={Zhou, Siyuan and Tan, Qida and Du, Wenchao and Chen, Hu and Yang, Hongyu},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={Learning Geometry-Aware Representation for Gaze Estimation}, 
  year={2025},
  volume={},
  number={},
  pages={1930-1935},
  keywords={Geometry;Adaptation models;Three-dimensional displays;Accuracy;Image processing;Estimation;Performance gain;Robustness;Faces;Gaze estimation;geometry-aware representation;normal map;cross-domain generalization},
  doi={10.1109/ICIP55913.2025.11084621}}
```
