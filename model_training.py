from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

def split_data(df):
    """
    Split the Formula 1 dataset into training and testing sets, with features and target variable separated.
    
    Features (X) include variables like grid position, driver experience, and position change.
    The target variable (y) is the final race finishing position.
    """
    X = df[['grid', 'position_change', 'is_fastest_lap', 'driver_experience', 'grid_experience_interaction', 'avg_position_last_5']]
    y = df['positionOrder']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model to predict Formula 1 race finishing positions.
    
    The Random Forest model is an ensemble of decision trees, which helps capture complex patterns in the data.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the Random Forest model's performance in predicting Formula 1 race results.
    
    The performance is measured using Mean Squared Error (MSE), which indicates how closely the model's predictions
    match the actual finishing positions.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest - Mean Squared Error: {mse}")

if __name__ == "__main__":
    file_path = 'data/results_features.csv'
    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
