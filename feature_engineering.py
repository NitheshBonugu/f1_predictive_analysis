import pandas as pd

def engineer_features(df):
    """
    Create new features from the Formula 1 race results data to improve model performance.
    
    Features include changes in race position, identification of fastest laps, driver experience,
    and interaction terms between these variables.
    """
    # Feature: Position Change (Grid - Final Position)
    df['position_change'] = df['grid'] - df['positionOrder']

    # Feature: Is Fastest Lap
    df['is_fastest_lap'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)

    # Feature: Driver Experience (assuming the dataset has raceId in chronological order)
    df['driver_experience'] = df.groupby('driverId').cumcount() + 1

    # Interaction Feature: Grid Position and Experience
    df['grid_experience_interaction'] = df['grid'] * df['driver_experience']

    # Rolling Average: Last 5 Races Average Finishing Position
    df['avg_position_last_5'] = df.groupby('driverId')['positionOrder'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # One-Hot Encoding: Convert categorical variables to numerical format
    df = pd.get_dummies(df, columns=['constructorId', 'statusId'], drop_first=True)

    return df

if __name__ == "__main__":
    file_path = 'data/results_cleaned.csv'
    df = pd.read_csv(file_path)
    df_features = engineer_features(df)
    df_features.to_csv('data/results_features.csv', index=False)
    print("Feature engineering completed. Enhanced data saved as 'results_features.csv'.")
