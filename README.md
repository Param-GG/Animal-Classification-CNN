# Animal-Classification-CNN
# Convolutional Neural Network for Dogs vs Cats Classification
# Overview
This repository contains a Google Colab file implementing a Convolutional Neural Network (CNN) based on the VGG architecture for the classification of images of dogs and cats. The model consists of 3 blocks with 32, 64, and 128 filters respectively.

# Dataset
The dataset used for training and testing the model is the Kaggle Dogs vs Cats dataset, consisting of 25,000 images in total, with 12,500 images of dogs and 12,500 images of cats.

# Model Architecture
The CNN model follows the VGG architecture, known for its simplicity and effectiveness in image classification tasks. The model is organized into three blocks, each comprising convolutional layers, followed by max-pooling layers for feature extraction.

Block 1: 32 filters
Block 2: 64 filters
Block 3: 128 filters
# Training
The model was trained using the entire dataset, with 12,500 images for each class. Training was performed using the Google Colab platform. The training process includes loading the dataset, preprocessing the images, defining the model architecture, compiling the model, and fitting it to the training data. The training was run for a sufficient number of epochs to achieve a satisfactory accuracy.

# Evaluation
The trained model achieved an accuracy of 74.6% on the test set, indicating its effectiveness in distinguishing between images of dogs and cats.

# Usage
To reproduce the results or further experiment with the model, follow these steps:

Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/dogs-vs-cats-cnn.git
Open the Jupyter notebook in Google Colab.

Run the cells in the notebook sequentially to load the dataset, define the model, train the model, and evaluate its performance.

Optionally, you can tweak hyperparameters, modify the architecture, or use transfer learning to improve the model further.

# Dependencies
Make sure you have the following dependencies installed:

TensorFlow
Keras
NumPy
Matplotlib
You can install these dependencies using the following:

bash
Copy code
pip install tensorflow keras numpy matplotlib

Feel free to explore, modify, and enhance the code as needed.
