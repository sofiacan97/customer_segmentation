import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids

def run_kmeans(n_clusters, df_cleaned: pd.DataFrame, df_scaled: pd.DataFrame) -> pd.DataFrame:
    kmeans = KMeans(n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_lables = kmeans.fit_predict(df_scaled)
    df_cleaned["clusters"] = cluster_lables
    return df_cleaned


def run_kmedoids(n_clusters, df_cleaned: pd.DataFrame, df_scaled: pd.DataFrame) -> pd.DataFrame:
    X_train, X_test = train_test_split(df_scaled, test_size=0.6, random_state=42)
    kmedoids = KMedoids(n_clusters, random_state=0)
    clusters_train = kmedoids.fit_predict(X_train)
    clusters_test = kmedoids.predict(X_test)


    df_train = df_cleaned.loc[X_train.index]  
    df_train['Cluster_Label'] = clusters_train
    df_test = df_cleaned.loc[X_test.index]
    df_test['Cluster_Label'] = clusters_test  


    cluster_labels = pd.concat([df_train['Cluster_Label'], df_test['Cluster_Label']], axis=0)
    df_cleaned['Cluster_Label'] = cluster_labels
    return df_cleaned