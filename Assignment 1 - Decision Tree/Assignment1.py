'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float/double/large


def get_entropy_of_dataset(df):
    entropy = 0
    target = df[[df.columns[-1]]].values
    _, counts = np.unique(target, return_counts=True)
    total_count = np.sum(counts)
    for freq in counts:
        temp = freq/total_count
        if temp != 0:
            entropy -= temp*(np.log2(temp))
    return entropy


'''Return entropy of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float/double/large


def get_entropy_of_attribute(df, attribute):
    attribute_values = df[attribute].values
    unique_attribute_values = np.unique(attribute_values)
    rows = df.shape[0]
    entropy_of_attribute = 0
    for current_value in unique_attribute_values:
        df_slice = df[df[attribute] == current_value]
        target = df_slice[[df_slice.columns[-1]]].values
        _, counts = np.unique(target, return_counts=True)
        total_count = np.sum(counts)
        entropy = 0
        for freq in counts:
            temp = freq/total_count
            if temp != 0:
                entropy -= temp*np.log2(temp)
        entropy_of_attribute += entropy*(np.sum(counts)/rows)
    return(abs(entropy_of_attribute))


'''Return Information Gain of the attribute provided as parameter'''
# input:int/float/double/large,int/float/double/large
# output:int/float/double/large


def get_information_gain(df, attribute):
    information_gain = 0
    entropy_of_attribute = get_entropy_of_attribute(df, attribute)
    entropy_of_dataset = get_entropy_of_dataset(df)
    information_gain = entropy_of_dataset - entropy_of_attribute
    return information_gain


''' Returns Attribute with highest info gain'''
#input: pandas_dataframe
#output: ({dict},'str')


def get_selected_attribute(df):

    information_gains = {}
    selected_column = ''

    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

    max_information_gain = float("-inf")
    for attribute in df.columns[:-1]:
        information_gain_of_attribute = get_information_gain(df, attribute)
        if information_gain_of_attribute > max_information_gain:
            selected_column = attribute
            max_information_gain = information_gain_of_attribute
        information_gains[attribute] = information_gain_of_attribute
    return (information_gains, selected_column)


'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
