import pandas as pd

def data_clean(df):
        df.columns = df.columns.str.strip()
        for column in df.columns:
            if column in ['Weight','HB','BP']:
                df[column].fillna(value=df[column].mean(), inplace=True)
                '''Best way to fill these columns is to  replace them with means'''
                
            elif column in ['Community','Delivery phase','IFA','Education']:
                df[column].fillna(value=df[column].mode()[0], inplace=True)
                '''Mode is the best way to fill all these columns because
                we can approximate that the missing values will be the ones which appear the most number of times'''
                
            else:
                df[column].fillna(value=df[column].median(), inplace=True)
                '''other columns we will just go with median'''
        return df

df=pd.read_csv("C:/Users/sreya/Desktop/PYTHONandCfiles/Machine Learning/Machine Intelligence 5th Sem/Assignment3/LBW_Dataset.csv")
data_clean(df)
df.to_csv("cleaned_data.csv")
