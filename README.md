# real-estate-multimodal-regression

A multimodal regression system that predicts residential property prices by combining structured housing data with satellite imagery.

## Overview
Traditional real estate valuation models rely only on tabular features such as square footage, number of bedrooms, and construction quality. This project extends that approach by incorporating satellite images to capture environmental and neighborhood context that is not present in structured data.

The final system integrates numerical features with visual features extracted from satellite imagery to improve price prediction performance.

## Satellite Imagery
Satellite images are programmatically collected using the Mapbox Static Images API, based on latitude and longitude coordinates provided in the dataset.

- One satellite image per unique property  
- Images capture surrounding neighborhood context  
- Images are aligned with tabular records using property IDs  

A pretrained ResNet-18 model is used as a fixed feature extractor to convert each image into a 512-dimensional embedding.

## Modeling Approach
- Tabular models: Linear Regression (baseline), Random Forest, and XGBoost  
- Image-only model: XGBoost trained on CNN embeddings  
- Multimodal model: Concatenation of tabular features and image embeddings, trained using XGBoost  

The multimodal model consistently matches or outperforms tabular-only models, demonstrating the value of visual information.

## Explainability
Model predictions are interpreted using Grad-CAM, which highlights the regions of satellite images that most influence price predictions. This provides insight into how environmental factors such as green spaces and infrastructure affect valuation.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, PyTorch, Torchvision, OpenCV, Matplotlib, Seaborn, Mapbox Static Images API

## Author
Rushita Guddeti
