# Predicting Formula 1 Race Results with Machine Learning

## Project Overview

This project aims to predict Formula 1 race finishing positions using various machine learning models. By leveraging historical race data, the project explores how different factors such as grid position, driver experience, and race conditions influence the final race results. The models are trained on historical data and evaluated based on their predictive accuracy, with the goal of identifying the best model for forecasting future race outcomes.

## Project Structure

- **`data_preparation.py`**: This script cleans the raw Formula 1 race data by handling missing values and ensuring the dataset is ready for further analysis.
- **`feature_engineering.py`**: This script creates new features that are believed to be important predictors of race outcomes, such as position changes, fastest laps, and driver experience.
- **`eda.py`**: This script performs exploratory data analysis to understand the structure, distributions, and relationships in the dataset, both before and after feature engineering.
- **`model_comparison.py`**: This script compares different machine learning models (Random Forest, Linear Regression, SVR, XGBoost) to determine which performs best in predicting race results.
- **`xgboost_tuning.py`**: This script performs hyperparameter tuning for the XGBoost model using GridSearchCV to find the optimal parameters that minimize prediction error.
- **`final_model_evaluation.py`**: This script evaluates the final XGBoost model using the best hyperparameters on the test set, providing detailed performance metrics and visualizations.

## Getting Started

### Prerequisites

Ensure that you have Python 3.6+ installed on your system. You will also need the following Python libraries, which can be installed using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
### Project Setup
#### Clone the Repository: 
Clone this repository to your local machine using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
#### Navigate to the Project Directory:
```bash 
cd formula1-race-prediction
```
#### Set Up the Environment (Optional but recommended): 
It's a good practice to create a virtual environment to manage dependencies:
```bash
python -m venv f1_env
source f1_env/bin/activate  
# On Windows use `f1_env\Scripts\activate`
```

## Running the Scripts

Follow these steps in order to clean the data, engineer features, and train models.


### 1. Data Preparation

#### Clean the raw data by running the `data_preparation.py` script. This will handle missing values and prepare the data for analysis.

```bash
python data_preparation.py
```
#### Output: A cleaned dataset saved as `results_cleaned.csv`.

### 2. Feature Engineering

#### Create additional features that could improve model performance using the `feature_engineering.py` script.
```bash
python feature_engineering.py
```
#### Output: An enhanced dataset with new features saved as `results_features.csv`.

### 3. Exploratory Data Analysis (Optional)

#### Understand the data distributions, structures, and relationships by running the `eda.py` script.
```bash
python eda.py
```
#### Output: Displays data summaries, statistics, and visualizations for both raw and feature-engineered datasets.

### 4. Model Comparison

#### Compare the performance of different machine learning models by running the `model_comparison.py` script.
```bash
python model_comparison.py
```
#### Output: Mean Squared Error (MSE) for each model, helping you identify the best-performing model.

### 5. Hyperparameter Tuning

#### Optimize the XGBoost model by running the `xgboost_tuning.py` script, which searches for the best set of hyperparameters.
```bash
python xgboost_tuning.py
```
#### Output: Best hyperparameters and the corresponding cross-validation score.

### 6. Final Model Evaluation

#### Evaluate the final XGBoost model using the best hyperparameters found during tuning.
#### This script will assess the model's performance on the test set and generate a plot comparing actual vs. predicted race positions.
```bash
python final_model_evaluation.py
```
#### Output: Mean Squared Error (MSE), R-squared score, and a plot of actual vs. predicted race positions.

### Results and Interpretation

## Key Metrics

### Mean Squared Error (MSE): 
Indicates how close the predicted race positions are to the actual positions on average. Lower MSE values indicate better model performance.

### R-squared Score (R²): 
Measures the proportion of variance in the race positions that can be explained by the model. Values closer to 1.0 indicate better explanatory power.

## Best Model

After comparing various models and tuning the XGBoost model, the final model's performance metrics are displayed in the `final_model_evaluation.py` output. This model is the best suited for predicting future Formula 1 race results based on the provided data.

## License

### This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.


