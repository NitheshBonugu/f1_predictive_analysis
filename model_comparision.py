from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df):
    """
    Split the Formula 1 dataset into training and testing sets, with features and target variable separated.
    
    Features (X) include variables like grid position, driver experience, and position change.
    The target variable (y) is the final race finishing position.
    """
    # Selecting relevant features for predicting race results
    X = df[['grid', 'position_change', 'is_fastest_lap', 'driver_experience', 'grid_experience_interaction', 'avg_position_last_5']]
    y = df['positionOrder']
    
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def try_different_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate different machine learning models to predict Formula 1 race results.
    
    Models tested include Random Forest, Linear Regression, Support Vector Regressor (SVR), and XGBoost.
    The performance of each model is compared using Mean Squared Error (MSE) to determine which
    model best predicts the race finishing positions.
    """
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regressor": SVR(kernel='rbf'),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        print(f"\nTraining {name} to predict Formula 1 race results...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} - Mean Squared Error: {mse}")

if __name__ == "__main__":
    # Load the feature-engineered Formula 1 dataset
    file_path = 'data/results_features.csv'
    df = pd.read_csv(file_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Train and evaluate different models
    try_different_models(X_train, X_test, y_train, y_test)
