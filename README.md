# CIFAR-10 Classification with a Modified ResNet-18

This project implements a **ResNet-18** variant for the CIFAR-10 dataset using PyTorch. The model has fewer than **5M** parameters (meeting the competition limit) and uses data augmentation techniques such as random cropping and horizontal flipping. It achieves around **90%+** validation accuracy on CIFAR-10.

## Project Overview

- **Dataset**: CIFAR-10  
- **Model**: Modified ResNet-18  
- **Key Features**:
  - Uses residual blocks (BasicBlock) for deeper yet trainable networks.
  - Parameter count is under 5 million, complying with competition requirements.
  - Provides end-to-end training scripts, inference scripts, and a visualization notebook.

## Main Files

- `notebooks/cifar10_resnet18.ipynb`  
  A Jupyter Notebook demonstrating the entire pipeline (data loading, model training, visualization, inference, etc.). You can run it directly to view training curves and other analyses.

- `src/dataset.py`  
  Helper functions and a custom `Dataset` class for loading and preprocessing the data.

- `src/model.py`  
  Contains the `BasicBlock`, `ResNet`, and `ResNet18` definitions.

- `src/train.py`  
  Implements the training loop, validation loop, and model checkpoint saving. It can be run from the command line.

- `src/inference.py`  
  Loads the best model checkpoint for inference on the test set and outputs the submission CSV file.

- `src/utils.py`  
  Auxiliary functions, such as plotting training curves or computing a confusion matrix (if needed).

## Environment Setup

This project uses Python 3.8+ with the following main dependencies:
numpy==1.23.5 pandas==1.5.3 torch==1.13.1 torchvision==0.14.1 tqdm==4.64.1 matplotlib==3.6.2

## Quick Start

### Download CIFAR-10 Dataset

This project uses the CIFAR-10 dataset provided by Kaggle. 

If you don't have a kaggle command line tool, please firstly run:
```
pip install kaggle
```

Then you need to create a new folder called `data`, download the data set and put it into the folder with the following code：
```
kaggle competitions download -c deep-learning-spring-2025-project-1
unzip deep-learning-spring-2025-project-1.zip -d data/
```

### Run the model

If you use **Jupyter Notebook**, please directly run the file: project-1-final.ipynb

If you use **any other Python IDE**, please download the files in `src` folder.
