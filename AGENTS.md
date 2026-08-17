# AI Agent Helper File (AGENTS.md)

## Project Context
**Project Name:** Multiclass Fish Image Classification
**Domain:** Deep Learning, Computer Vision, Web Application

## Goals & Objectives
*   Build a CNN from scratch to classify fish images.
*   Fine-tune pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0) for the same task.
*   Compare models and save the best one as an `.h5` file.
*   Develop a Streamlit application to allow users to upload images and receive predictions.

## Architecture
*   **Model Training Environment:** Google Colab (Primary).
*   **Local Application Environment:** Streamlit (Local testing and deployment).
*   **Version Control:** Git & GitHub. Keep the repository public.
*   **Coding Standards:** Follow PEP-8 standard conventions.

## Instructions for AI Agents
1.  **Code Consistency:** Ensure all Python code, especially Streamlit app and utility functions, are modular and follow PEP-8.
2.  **Git Commits:** Commit frequently using clear and descriptive messages. Use conventional commit prefixes (e.g., `feat:`, `fix:`, `docs:`, `chore:`).
3.  **Documentation:** Keep `README.md` updated as the project evolves. Document approach, code, and evaluation metrics clearly.
4.  **Environment Management:** Ensure the local virtual environment dependencies are tracked in `requirements.txt`.
5.  **Data Handling:** Always perform validation on data. Assume dataset is provided in a zip file.
6.  **Deliverables:** Pay attention to deliverables like Jupyter Notebooks for model training and Python scripts for Streamlit deployment.
