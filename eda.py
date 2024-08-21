import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def initial_eda(df):
    """
    Perform initial exploratory data analysis on the raw Formula 1 race results dataset.
    
    This step provides insights into the basic structure of the data, distributions, and potential issues
    such as missing values.
    """
    print("Initial Data Information:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    # Example plots for understanding the distribution of race positions
    sns.histplot(df['positionOrder'], bins=20)
    plt.title('Distribution of Final Race Positions')
    plt.show()

    # Only keep numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=[float, int])
    
    # Drop columns that have any NaN values (since these can't be included in a correlation matrix)
    numeric_df = numeric_df.dropna(axis=1, how='any')
    
    # Calculate and plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Initial Data)')
    plt.show()

def post_feature_engineering_eda(df):
    """
    Explore the Formula 1 dataset after feature engineering has been applied.
    
    This step allows you to assess the impact of new features on the data's structure and relationships.
    """
    print("Post-Feature Engineering Data Information:")
    print(df.info())
    
    print("\nSummary Statistics (After Feature Engineering):")
    print(df.describe())
    
    # Example plot for new features such as position change
    sns.histplot(df['position_change'], bins=20)
    plt.title('Distribution of Position Changes')
    plt.show()

    # Only keep numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=[float, int])
    
    # Drop columns that have any NaN values (since these can't be included in a correlation matrix)
    numeric_df = numeric_df.dropna(axis=1, how='any')
    
    # Calculate and plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (After Feature Engineering)')
    plt.show()

if __name__ == "__main__":
    # Initial EDA on raw data
    raw_file_path = 'data/results.csv'
    df_raw = pd.read_csv(raw_file_path)
    initial_eda(df_raw)

    # EDA after feature engineering
    feature_file_path = 'data/results_features.csv'
    df_features = pd.read_csv(feature_file_path)
    post_feature_engineering_eda(df_features)
