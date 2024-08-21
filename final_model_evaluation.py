import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from data_preparation import split_data

def load_data(file_path):
    """Load the Formula 1 dataset from the specified file path."""
    return pd.read_csv(file_path)

def evaluate_final_model(X_train, X_test, y_train, y_test, best_params):
    """
    Evaluate the final XGBoost model using the best hyperparameters on the test set.
    
    Parameters:
    - X_train: Training data features (e.g., grid position, driver experience).
    - X_test: Testing data features.
    - y_train: Training data target variable (e.g., race finishing position).
    - y_test: Testing data target variable.
    - best_params: The best hyperparameters found during tuning for the XGBoost model.
    
    Returns:
    - mse: Mean Squared Error on the test set.
    - r2: R-squared score on the test set.
    - y_pred: Predicted finishing positions for the test set.
    """
    # Initialize the XGBoost regressor with the best found hyperparameters
    model = XGBRegressor(**best_params, random_state=42)
    
    # Train the model on the entire training set
    model.fit(X_train, y_train)
    
    # Predict race finishing positions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, y_pred

def plot_results(y_test, y_pred):
    """
    Plot the actual vs. predicted race finishing positions to visually assess the model's performance.
    
    Parameters:
    - y_test: Actual race finishing positions.
    - y_pred: Predicted race finishing positions by the model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Finishing Position')
    plt.ylabel('Predicted Finishing Position')
    plt.title('Actual vs Predicted Finishing Positions')
    plt.show()

if __name__ == "__main__":
    # Load the feature-engineered Formula 1 dataset
    file_path = 'data/results_features.csv'
    df = load_data(file_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Best hyperparameters found during tuning for predicting race results
    best_params = {
        'learning_rate': 0.2,  # Controls how much the model learns in each boosting round
        'max_depth': 3,        # Maximum depth of each decision tree in the model
        'n_estimators': 200,   # Number of trees in the boosting process
        'subsample': 0.6       # Proportion of data used to train each tree (helps prevent overfitting)
    }

    # Evaluate the final model's ability to predict race finishing positions
    model, mse, r2, y_pred = evaluate_final_model(X_train, X_test, y_train, y_test, best_params)
    
    # Print the evaluation metrics
    print(f"Mean Squared Error on Test Set: {mse}")
    print(f"R-squared Score on Test Set: {r2}")

    # Plot the actual vs. predicted race finishing positions
    plot_results(y_test, y_pred)
