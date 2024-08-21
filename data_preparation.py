import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset from the specified file path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values and removing duplicates."""
    df.replace({'\\N': np.nan}, inplace=True)
    df_cleaned = df.dropna().drop_duplicates()
    df_cleaned['milliseconds'] = pd.to_numeric(df_cleaned['milliseconds'], errors='coerce')
    return df_cleaned

def split_data(df):
    """Split the dataset into features (X) and target (y), then into training and testing sets."""
    X = df[['grid', 'position_change', 'is_fastest_lap', 'driver_experience', 'grid_experience_interaction', 'avg_position_last_5']]
    y = df['positionOrder']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = 'data/results.csv'
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    df_cleaned.to_csv('data/results_cleaned.csv', index=False)
    print("Data cleaning completed.")
