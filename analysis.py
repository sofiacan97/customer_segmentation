import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def age_distribution(df: pd.DataFrame):
    cluster_colors = {0: 'yellow', 1: 'red', 2: 'blue'}
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Age'], fill=True, color='gray', label='Mean')
    for cluster in df['clusters'].unique():
        persona_data = df[df['clusters'] == cluster]
        sns.kdeplot(persona_data['Age'], fill=True, label=f'{cluster}', color=cluster_colors[cluster])

    plt.title('Age Distribution by cluster')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend(fontsize='large')
    plt.show()

def language_distribution(df: pd.DataFrame):
    cluster_colors = {0: 'yellow', 1: 'red', 2: 'blue'}
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Language', hue='clusters', palette=cluster_colors)
    plt.title('Language Distribution by Clusters')
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.show()

def frequency_AOV_distribution(df: pd.DataFrame):
    cluster_colors = {0: 'yellow', 1: 'red', 2: 'blue'}
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='frequency', y='AOV', hue='clusters', style='clusters', palette=cluster_colors)
    plt.title('Frequency vs AOV Distribution by Clusters')
    plt.xlabel('Frequency')
    plt.ylabel('AOV')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()