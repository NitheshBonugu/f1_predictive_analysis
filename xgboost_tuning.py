import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from data_preparation import split_data

def load_data(file_path):
    """Load the Formula 1 dataset from the specified file path."""
    return pd.read_csv(file_path)

def tune_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning on the XGBoost model using GridSearchCV.
    
    The goal is to find the best set of hyperparameters that minimize the prediction error
    when estimating Formula 1 race finishing positions.
    """
    # Define the parameter grid to search over
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of boosting rounds (trees)
        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
        'max_depth': [3, 5, 7],  # Maximum depth of each decision tree
        'subsample': [0.6, 0.8, 1.0]  # Fraction of data used to train each tree
    }

    # Initialize the XGBoost regressor
    xgb = XGBRegressor(random_state=42)

    # Set up GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Return the best model and the results of the grid search
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

if __name__ == "__main__":
    # Load the feature-engineered Formula 1 dataset
    file_path = 'data/results_features.csv'
    df = load_data(file_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Perform hyperparameter tuning for the XGBoost model
    best_model, best_params, best_score = tune_xgboost(X_train, y_train)

    # Print the best parameters and the corresponding cross-validation score
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score (negative MSE): {-best_score}")
