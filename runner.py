import pandas as pd
from eda import categorical_eda, numerical_eda
from scaling import scaling
from evaluation_visualization import kmeans_elbow, kmedoids_elbow 
from clustering import run_kmeans, run_kmedoids
from analysis import age_distribution

def data_processing(df: pd.DataFrame) -> pd.DataFrame:
    """take all Retail loyalty customers with company italy and a valid age"""
    df = df.loc[(df["Company"] == 'IT1') & (df["Loyalty"] == 'Yes') 
        & (df["Age"] >= 18) & (df["Age"] <= 90) & (df["Client Type"] == 'Retail')]
    df_cleaned = df.drop(['Company','Loyalty', 'CustomerID', 'Client Type'], axis=1)
    return df_cleaned

if __name__ == "__main__":
    # Read CSV file
    df = pd.read_csv('customer_segmentation/customers.csv')

    # Data processing
    df_processed = data_processing(df)

    # Exploratory Data Analysis
    categorical_eda(df_processed)
    numerical_eda(df_processed)
    
    # scaling
    df_scaled = scaling(df_processed)

    kmeans_elbow(df_scaled)
    # kmedoids_elbow(df_scaled) 

    df_clustered = run_kmeans(3, df_processed, df_scaled)
    # df_clustered = run_kmedoids(4, df_scaled)
    print(df_clustered.head())

    # analysis example on the clustered customers
    age_distribution(df_clustered)