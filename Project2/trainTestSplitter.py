import pandas as pd
import numpy as np

def collectData(filePath):
    return pd.read_excel(filePath)

def removeValidationData(df):
    newDF = df.copy()
    newDF.drop(newDF[newDF['Training / Validation'] == 'Validation'].index, inplace = True)
    return newDF

def removeTrainingData(df):
    newDF = df.copy()
    newDF.drop(newDF[newDF['Training / Validation'] == 'Training'].index, inplace = True)
    return newDF

def filterCols(df, targetVar):
    colNames = ['Width', 'Height', 'Width_Variance', 'Height_Variance']
    if isinstance(targetVar, list):
        for i in targetVar:
            colNames.remove(i)
    else:
        colNames.remove(targetVar)
    df = df.drop(columns=colNames)
    return df

def removeUnwantedCols(df):
    return df.drop(columns=['ID', 'Training / Validation', 'Wire', 'Substrate Type'])

def splitTargetVariables(df):
    datasets = {}
    datasets['Width'] = filterCols(df, 'Width')
    datasets['Height'] = filterCols(df, 'Height')
    datasets['Width_Variance'] = filterCols(df, 'Width_Variance')
    datasets['Height_Variance'] = filterCols(df, 'Height_Variance')
    datasets['Width_Height'] = filterCols(df, ['Width', 'Height'])
    return datasets

def partitionData(df, num_partitions):
    df_shuffled = df.sample(frac=1, random_state=42)  
    partition_size = len(df_shuffled) // num_partitions
    partitions = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i < num_partitions - 1 else len(df_shuffled)
        partition = df_shuffled.iloc[start_idx:end_idx]
        partitions.append(partition)
    return partitions

if __name__ == '__main__':
    df = collectData('../training/FINALIZED_DATASET.xlsx')
    trainingDF = removeValidationData(df)
    testingDF = removeTrainingData(df)
    # print(trainingDF)
    # print(testingDF)
    print(splitTargetVariables(trainingDF))
