# World Happiness Report 2020 - Machine Learning Models

This project uses the [World Happiness Report 2020 dataset](https://www.kaggle.com/) from Kaggle to predict happiness scores based on various input features. The goal is to compare the performance of four different machine learning algorithms and predict whether a given country is considered "low happy", "medium happy", or "high happy".

## Algorithms Implemented
- **Linear Regression**
- **Decision Tree Regressor**
- **Gradient Boosting Regressor**
- **Random Forest Regressor**

Additionally, the project provides an interactive Streamlit application where users can input feature values and predict the happiness category.

## Features
- **Model Evaluation**: Calculates and compares the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for each model.
- **Plots**: Visualizes model performance through various plots.
- **SQL Queries**: Executes 10 analytic queries using SQL on the dataset.
- **User Input**: The application allows users to input values for different features and predicts whether a country falls into a "low happy", "medium happy", or "high happy" category.

## Dataset
The project uses the World Happiness Report 2020 dataset from Kaggle. You can download it from [here](https://www.kaggle.com/datasets/mllion/world-happiness-report-2024).

## Technologies Used
- **Machine Learning Algorithms**: Scikit-learn (Linear Regression, Decision Tree, Gradient Boosting, Random Forest)
- **Web Application**: Streamlit
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib
- **SQL Database**: SQLite3 (for analytic queries)

## Installation

Follow the steps below to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```
### 2. Install Dependencies
You will need Python (preferably version 3.7 or later). You can install the required libraries using pip:
```bash
pip install streamlit matplotlib numpy pandas scikit-learn
```
### 3. Running the Application
Once the dependencies are installed and the dataset is in place, you can run the Streamlit application by using the following command:

```bash

streamlit run new.py
```
This will launch the app in your default web browser.
