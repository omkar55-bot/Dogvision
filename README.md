# Dog Vision: Dog Breed Classification Project

## Overview
This project aims to classify images of dogs into 120 different breeds using a deep learning model. The project leverages transfer learning with **MobileNetV2** and **NasNetLarge** (both pre-trained on ImageNet) to achieve high accuracy. The experiments also involve data augmentation to improve model generalization.

## Dataset
- **Source**: [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/competitions/dog-breed-identification/data)
- The dataset includes:
  - **Train Images**: 10,000+ labeled images (120 breeds).
  - **Test Images**: 10,000+ unlabeled images for evaluation.
  - **Labels.csv**: File containing breed labels for training images.

## Project Goals
- Implement a deep learning pipeline using transfer learning.
- Perform data augmentation to improve generalization.
- Compare the performance of different architectures (MobileNetV2 and NasNetLarge) on augmented datasets.

## Data Preprocessing
- **Batch Creation**: A function was created to generate batches of data for training, validation, and testing. This ensures that the data is efficiently processed in batches, making use of TensorFlow's `tf.data.Dataset` for better performance.
- **Image Processing**: Images were preprocessed by reading the image files, decoding them into tensors, normalizing the pixel values, and resizing them to a fixed size of 224x224 pixels (as required by the models).
- **Label Association**: For each image file, its corresponding label was retrieved, and the images and labels were returned as tuples for model training and evaluation.

## Model Architectures
- **MobileNetV2**: Lightweight model optimized for mobile and efficient inference.
- **NasNetLarge**: A deeper model with more capacity, designed for better generalization.

## Training and Evaluation
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Additional Techniques**:
  - Early stopping to prevent overfitting.
  - Data augmentation, including random rotations and flips, to increase training robustness using tensorflow.Keras.Sequential Layers.

## Experimentation and Results

### **1. Experiment 1: 10,000 Augmented Images**
- **MobileNetV2**:
  - **Training Accuracy**: 99.8%
  - **Validation Accuracy**: 80.25%

- **NasNetLarge**:
  - **Training Accuracy**: 98%
  - **Validation Accuracy**: 93.7%

### **2. Experiment 2: Full Augmented Dataset (18,222 Images)**
- **MobileNetV2**:
  - **Training Accuracy**: 97.3%

- **NasNetLarge**:
  - **Training Accuracy**: 97.1%

### **Conclusion**:
The experiments demonstrate that **NasNetLarge** generalizes better on the validation set compared to **MobileNetV2**. Despite having slightly lower training accuracy, **NasNetLarge** consistently outperforms in validation, indicating better resistance to overfitting.

## Directory Structure
```
Dog-Breed-Classification/
├── Data/                   # Contains the dataset (train and test images)
├── custom_images/          # Custom images for testing
├── models/                 # Saved model weights and .h5 files
├── logs/                   # Training and evaluation logs
├── dog_vision.ipynb        # Main Jupyter notebook
├── requirements.txt        # Python dependencies
├── README.md               # Project description
```
