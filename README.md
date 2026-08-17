# Multiclass Fish Image Classification

## Problem Statement
This project focuses on classifying fish images into multiple categories using deep learning models. The task involves training a CNN from scratch and leveraging transfer learning with pre-trained models to enhance performance. The trained models will be deployed using a Streamlit application to predict fish categories from user-uploaded images.

## Business Use Cases
*   **Enhanced Accuracy**: Determine the best model architecture for fish image classification.
*   **Deployment Ready**: Create a user-friendly web application for real-time predictions.
*   **Model Comparison**: Evaluate and compare metrics across models to select the most suitable approach for the task.

## Approach
1.  **Data Preprocessing and Augmentation**:
    *   Rescale images to `[0, 1]` range.
    *   Apply data augmentation techniques like rotation, zoom, and flipping to enhance model robustness.
2.  **Model Training**:
    *   Train a CNN model from scratch.
    *   Experiment with five pre-trained models: VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0.
    *   Fine-tune the pre-trained models on the fish dataset.
    *   Save the best performing model in `.h5` format for future use.
3.  **Model Evaluation**:
    *   Compare metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
    *   Visualize training history (accuracy and loss).
4.  **Deployment**:
    *   Build a Streamlit application to allow users to upload fish images, predict the fish category, and provide model confidence scores.

## Dataset
Dataset is available in the form of a Zip file containing images categorized into folders by species.

## Project Deliverables
*   **Trained Models**: CNN and pre-trained models saved in `.h5`.
*   **Streamlit Application**: Interactive web app for real-time predictions.
*   **Jupyter Notebooks / Python Scripts**: For training, evaluation, and deployment (Training will be mostly done in Google Colab).
*   **Comparison Report**: Metrics and insights from all models.
*   **GitHub Repository**: Well-documented codebase.

## Workflow and Execution
1.  Clone this repository.
2.  Extract the dataset and place it in the designated folder.
3.  Train the model in Google Colab (using the provided notebooks) and save the `.h5` model file.
4.  Download the trained `.h5` model to the local repository.
5.  Set up the local environment by installing the dependencies from `requirements.txt`.
6.  Run the Streamlit application using `streamlit run app.py`.

## Coding Standards
Follow PEP-8 guidelines for Python code.
