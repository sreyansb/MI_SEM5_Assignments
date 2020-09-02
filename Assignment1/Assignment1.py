<<<<<<< HEAD
'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

#to get all the values in the last column of the data frame

def get_all_values_of_final_output(df):
    s=set()
    for i in df.iloc[:,-1]:
        if i not in s:
            s.add(i)
    return s

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

#Formula -> For all outputs present in the last column of the dataframe
# apply -(k*log2(k))
# where k=Count of each output/length of data frame
def get_entropy_of_dataset(df):
    entropy = 0
    di={}
    total=0
    for i in df.iloc[:,-1]:#all the rows for the last column in the dataframe
        if i not in di:
            di[i]=0
        di[i]+=1
        total+=1
    if total!=0:
        for i in di:
            entropy+=-((di[i]/total)*np.log2(di[i]/total))          
    return entropy

'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large

#Formula -> For all outputs present in that attribute's column of the dataframe
#We need to get the ratio of how many of these O/Ps provide each of the outputs
#of the last column and for each of the outputs of the last columns get pilog2(pi)
#if pi=0 then skip it.
#average info of an attribute might be entropy and you multiply the previous
#step answer to (total of current output attribute)/(total # of records in the data frame)
def get_entropy_of_attribute(df,attribute):
    
    entropy_of_attribute = 0
    di={}
    totalfinal=0
    k=df[attribute]
    s=get_all_values_of_final_output(df)
    for i in range(len(k)):
        if k[i] not in di:
            di[k[i]]={}
            for all_values in s:
                di[k[i]][all_values]=0                
        di[k[i]][df.iloc[i][-1]]+=1
        totalfinal+=1
    for i in di:
        total=0
        for j in di[i]:
            total+=di[i][j]
        if total!=0:
            for j in s:
                if di[i][j]!=0:
                    k=di[i][j]/total
                    entropy_of_attribute+=-(k)*np.log2(k)*(total/totalfinal)
    return abs(entropy_of_attribute)

'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large

def get_information_gain(df,attribute):
	information_gain = 0
	datasetentropy=get_entropy_of_dataset(df)
	attributeentropy=get_entropy_of_attribute(df,attribute)
	information_gain=datasetentropy-attributeentropy
	return information_gain

''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')

def get_selected_attribute(df):
    information_gains={}
    selected_column=''
    maxi=-1 #since no info gain can ever be negative
    for attribute in df.columns[:-1]:
        information_gains[attribute]=get_information_gain(df,attribute)
        if information_gains[attribute]>maxi:
            maxi=information_gains[attribute]
            selected_column=attribute
    return (information_gains,selected_column)  

    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
        
    



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
=======
'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

#to get all the values in the last column of the data frame

def get_all_values_of_final_output(df):
    s=set()
    for i in df.iloc[:,-1]:
        if i not in s:
            s.add(i)
    return s

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

#Formula -> For all outputs present in the last column of the dataframe
# apply -(k*log2(k))
# where k=Count of each output/length of data frame
def get_entropy_of_dataset(df):
    entropy = 0
    di={}
    total=0
    for i in df.iloc[:,-1]:#all the rows for the last column in the dataframe
        if i not in di:
            di[i]=0
        di[i]+=1
        total+=1
    if total!=0:
        for i in di:
            k=di[i]/total
            if k!=0:
                entropy+=-((k)*np.log2(k))          
    return entropy

'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large

#Formula -> For all outputs present in that attribute's column of the dataframe
#We need to get the ratio of how many of these O/Ps provide each of the outputs
#of the last column and for each of the outputs of the last columns get pilog2(pi)
#if pi=0 then skip it.
#average info of an attribute might be entropy and you multiply the previous
#step answer to (total of current output attribute)/(total # of records in the data frame)
def get_entropy_of_attribute(df,attribute):
    
    entropy_of_attribute = 0
    di={}
    totalfinal=0
    k=df[attribute]
    s=get_all_values_of_final_output(df)
    for i in range(len(k)):
        if k[i] not in di:
            di[k[i]]={}
            for all_values in s:
                di[k[i]][all_values]=0                
        di[k[i]][df.iloc[i][-1]]+=1
        totalfinal+=1
    for i in di:
        total=0
        for j in di[i]:
            total+=di[i][j]
        for j in s:
            if di[i][j]!=0:
                k=di[i][j]/total
                entropy_of_attribute+=-(k)*np.log2(k)*(total/totalfinal)
    return abs(entropy_of_attribute)

'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large

def get_information_gain(df,attribute):
	information_gain = 0
	datasetentropy=get_entropy_of_dataset(df)
	attributeentropy=get_entropy_of_attribute(df,attribute)
	information_gain=abs(datasetentropy-attributeentropy)
	return information_gain

''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')

def get_selected_attribute(df):
    information_gains={}
    selected_column=''
    maxi=-1 #since no info gain can ever be negative
    for attribute in df.columns[:-1]:
        information_gains[attribute]=get_information_gain(df,attribute)
        if information_gains[attribute]>maxi:
            maxi=information_gains[attribute]
            selected_column=attribute
    return (information_gains,selected_column)  

    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
        
    



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
>>>>>>> 8be07a999f3abf4318620c099d47c6196f992b7e
