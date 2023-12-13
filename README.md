# Tracing-Human-Imprint
Utilizing deep learning on satellite images to categorize land cover features, aiding land management decisions and enhancing security by accurately identifying ground cover and infrastructure in various regions.
Using Deep Learning to Segment Land Types in Satellite Images

This project aims to develop a deep learning model for segmenting land types in satellite images using the DeepGlobe Land Cover Classification dataset available on Kaggle. The project uses the segmentation_models library to select and train models for image segmentation.

## Dataset
The DeepGlobe Land Cover Classification dataset contains satellite images of different parts of the world and corresponding segmentation masks. The dataset consists of 6,115 training images and 2,558 testing images with a resolution of 2448x2448 pixels. The dataset is available on Kaggle and can be downloaded using the following link: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

## Requirements
The project requires the following dependencies:

- Python 3.6 or higher
- Keras 2.2.5 or higher
- segmentation_models 1.0.1 or higher
- matplotlib 3.3.2 or higher
- numpy 1.19.2 or higher
- OpenCV 4.2.0 or higher
- pandas 1.1.3 or higher
- Pillow 8.3.1 or higher
- streamlit 0.86.0 or higher

## File Description
- app.py: This file contains the code for a web application that allows users to upload an image and see its land type segmentation.

- predict.py: This file contains functions to preprocess the image and apply the segmentation model.

- models: This folder contains pre-trained models for segmentation.

- README.md: This file contains information about the project.

# Usage
To run the web application, run the app.py file using the following command:
```sh
streamlit run app.py
```
The application will open in a web browser, and users can upload an image to see its land type segmentation.
