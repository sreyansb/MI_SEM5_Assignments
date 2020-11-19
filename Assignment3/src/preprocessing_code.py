import pandas as pd

def preprocess_data(df):
	#Clean data
	#Remove trailing blank spaces in column names
	df.columns = df.columns.str.strip()
	for column in df.columns:
	    if column in ['Weight','HB','BP']:
	        #Fill null values with mean for these columns
	        df[column].fillna(value=df[column].mean(), inplace=True)
	    elif column in ['Community','Delivery phase','IFA','Education','Result']:
	        #Fill null values with mode for these columns
	        df[column].fillna(value=df[column].mode()[0], inplace=True)
	    else:
	        #Fill null values with median for age
	        df[column].fillna(value=df[column].median(), inplace=True)
	return df

def main():
	#Read CSV file
    df = pd.read_csv('LBW_Dataset.csv')
    #Preprocess and save data 
    df = preprocess_data(df)
    df.to_csv('PP_LBW_Dataset.csv')
main()