# Gender privacy angular constraints for face recognition

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)](https://www.python.org/downloads/)

## Description

This repo contains the code to reproduce the results for the paper 'Gender privacy angular constraints for face recognition'.

## Table of Contents


- [Gender privacy angular constraints for face recognition](#gender-privacy-angular-constraints-for-face-recognition)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data](#data)
  - [Model](#model)
  - [Training](#training)
  - [Evaluate](#evaluate)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)

## Installation

Provide instructions on how to install and set up the project. Include any dependencies and versions needed.

```bash
pip install -r requirements.txt
```
## Data
The datasets LFW, AgeDB, ColorFeret and a sample of VGGFace2 are used in this project.
- [] We provide the verification pairs for ColorFeret and VGGFace2. 
- As for AgeDB and LFW, the standard verification pairs are used according to AgeDB-30 partition and the view 1 of the LFW verification protocole.

## Model
All the finetuning training is performed on the iResNet50 architecture pretrained with an ArcFace loss on the VGGFace2 dataset.  
- [] We provide the weights model finetuned on AgeDB with the privacy constraint $L_{p} = 20 L{p_1} + L{p_2}$.


## Training
After specifying the training configurations in the file [./config/config.py], run this command:

```bash
python finetune.py
```
Please refer to our paper for the choice of hyperparameters.
## Evaluate
```bash
python eval/evaluation.py --log_root "path/to/log" --model_root "path/to/model/directory"  --experiment-name "experimentname" --reference_pth "path/to/reference/csv/results" --pretrained_pth "path/to/pretrained/model/weights" --ft False --gpu-id 0

```
```experiment-name``` should refer to the different types of training settings. For instance if you train using only one dataset e.g ColorFeret, ```experiment-name``` = ```OneDataset/ColorFeret```. The script ```evaluate.py``` will run the evaluations of all the models generated with this experiment setting.

## Contributing

Contributions to this project are welcome. If you encounter any issues, have suggestions, or want to contribute improvements, please submit a pull request.

## Acknowledgments
 This repository contains some code from [InsightFace: 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface). If you find this repository useful for you, please also consider citing the paper [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://ieeexplore.ieee.org/document/8953658) paper!

