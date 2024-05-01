import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scaling(df: pd.DataFrame) -> pd.DataFrame:
    """scaling function using the standar scaler for numerical features and minMax scaler for ordinal categorical features"""
    new_df = df.copy()

    categorical_columns = new_df.select_dtypes(include=['object']).columns
    numerical_columns = new_df.select_dtypes(include=['int64', 'float64']).columns
    
    #numerical
    scaler = StandardScaler()
    scaler.fit(new_df[numerical_columns])

    numerical_data = scaler.transform(new_df[numerical_columns])
    numerical_data = pd.DataFrame(numerical_data,index=new_df.index,columns=numerical_columns)

    #ordial features
    ordinal_columns = ['Tier', 'TopFamily']
    
    new_df['Tier'] = new_df['Tier'].map({'GREEN':0, 'PINK':1, 'RED':2,'GOLD':3})
    new_df['TopFamily'] = new_df['TopFamily'].map({'CHOCOLATE':0, 'GELATO':1, 'BAR-COFFEE':2})

    ordinal_data = new_df[ordinal_columns]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(new_df[ordinal_columns])
    ordinal_data = scaler.transform(new_df[ordinal_columns])
    ordinal_data= pd.DataFrame(ordinal_data,index=new_df.index,columns=ordinal_columns)

    #nominal features
    nominal_columns = [c for c in categorical_columns if c not in ordinal_columns]

    nominal_data = list()
    for i,x in df[nominal_columns].nunique().items():
        if x <= 2:
            nominal_data.append(pd.get_dummies(new_df[[i]],drop_first=True))
        elif x > 2:
            nominal_data.append(pd.get_dummies(new_df[[i]],drop_first=False))       
    nominal_data = pd.concat(nominal_data,axis=1)

    #final dataset
    return  pd.concat([numerical_data, ordinal_data, nominal_data], axis=1)