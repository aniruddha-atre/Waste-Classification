# Background & Purpose

There is a global concern over rising beach pollution. Manual waste collection is a tedious task. This project is a part of Beach CLeaning project, that can automatically detect and classify waste through live feed. Thus, the purpose of this project is to implement a Deep Learning model for waste classification with a high accuracy.

## The Data

The dataset used is available on Kaggle and has over 15000 images and is divided into 12 classes. 

Link to data: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

For the purpose of this project, I have taken the following 5 classes - Cardboard, Glass,  Metal, Paper, Plastic
The data is structured into three subfolders: "train", "val", and "test", and within each, there are five folders: "cardboard", "glass", "metal", "paper", "plastic"

Train: Used for model training with data augmentation.
Validation: Used to monitor performance and adjust learning rate.
Test: Used for final evaluation and performance analysis.

Data augmentation techniques such as random horizontal flipping, rotation, and color jittering are applied to enhance model generalization.

## Model Training and Evaluation

I have used transfer learning, with the pretrained weights of MobileNetV2 model pretrained on ImageNet data. The model is fine-tuned on the custom dataset for classifying waste into different categories using the above dataset. The system is implemented in PyTorch and includes features like data augmentation, learning rate scheduling, training visualization, and performance evaluation using a confusion matrix.

## Model Architecture

1) Base Model: MobileNetV2 (pretrained on ImageNet)
2) Classifier: A fully connected layer replacing the original classifier
3) Loss Function: Cross-Entropy Loss
4) Optimizer: Adam optimizer with weight decay
5) Scheduler: Learning rate reduction on validation loss plateau
  
## Results

The model is trained for 100 epochs using a batch size of 32. 
Learning rate scheduling is applied to optimize convergence.
The training loss decreases steadily, while the validation loss stabilizes after around 30 epochs.
The accuracy improves significantly in early epochs and stabilizes at around 97% validation accuracy.

![image](https://github.com/user-attachments/assets/3f81ccb8-99c3-4de0-9f62-dc47c9b928f2)

![image](https://github.com/user-attachments/assets/39800271-e298-4ebd-a051-3efd2c8fb6d7)

![image](https://github.com/user-attachments/assets/0086e0d5-d60f-49d8-8b68-56dadc69eff4)

## Conclusions
This project demonstrates a robust approach to waste classification using MobileNetV2 and transfer learning. The model achieves high accuracy with good generalization. This is a small demo in applying machine learning methods to wasrw classification and recycling. In order to make a significant impact in waste sorting plants and on the recycling rate globally, much more sophisticated models, i.e., ones that can classify materials into multiple categories and sub-categories, would need to be developed so that waste could be efficiently and correctly identified at a large scale level.

## Prerequisites

Before running the project, ensure everything is installed:

Setup

Clone the repository:

git clone https://github.com/aniruddha-atre/Waste-Classification

Install dependencies:

pip install -r requirements.txt

Prepare the dataset and place it in the data/ folder.

Update the path to the data folder on your local machine

Run the Jupyter notebook
