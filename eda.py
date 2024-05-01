import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def categorical_eda(df: pd.DataFrame):
    categorical_columns = df.select_dtypes(include=['object']).columns

    num_plots = len(categorical_columns)
    num_cols = 4  
    num_rows = -(-num_plots // num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    for i, column in enumerate(categorical_columns):
        sns.countplot(x=column, data=df, ax=axes[i])
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')

    for j in range(i+1, num_cols*num_rows):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def numerical_eda(df: pd.DataFrame):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    num_plots = len(numerical_columns)
    num_cols = 4  
    num_rows = -(-num_plots // num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column], ax=axes[i])
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')

    for j in range(i+1, num_cols*num_rows):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
