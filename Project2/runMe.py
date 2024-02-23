import pandas as pd
import numpy as np
import models, trainTestSplitter, eval
import warnings
warnings.filterwarnings('ignore')

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

def move_specific_strings_to_end(input_list):
    strings_to_keep, strings_to_look_for, strings_to_move  = [], ["Width", "Height", "Width_Variance", "Height_Variance"], []

    for string in input_list:
        if string in strings_to_look_for:
            strings_to_move.append(string)
        else:
            strings_to_keep.append(string)

    result_list = strings_to_keep + strings_to_move

    return result_list


def removeUnwantedCols(df):
    return df.drop(columns=['ID', 'Training / Validation', 'Wire', 'Substrate Type'])

def removeUnwantedContCols(df):
    columns_to_keep = [col for col in df.columns if not (col.startswith('S_') or col.startswith('W_')
                                                         or col.startswith('Training') or col.startswith('ID'))]

    # Create a new DataFrame with only the selected columns
    df_filtered = df[columns_to_keep]

    df_encoded = pd.get_dummies(df_filtered, columns=['Substrate Type', 'Wire'])
    colOrder = move_specific_strings_to_end(df_encoded.columns)
    df_encoded = df_encoded[colOrder]
    return df_encoded

def runModels(df, NUM_PARTITIONS = 5, POLY_REG_DEGREE = 2, RIDGE_REG_ALPHA = 1, LASSO_ALPHA = 1, LASSO_INTERCEPT_FIT = True, 
              ELASTIC_ALPHA=1, ELASTIC_L1=0, ELASTIC_SHOULD_FIT=1, ELASTIC_SELECTION='cyclic', KNN_CLUSTERS = 5, DT_MAX_DEPTH = 10, DT_MIN_SAMPLE_SPLIT = 2,
              PCR_PARAMETERS = 2, PCR_DEGREE = 3,
              DT_CRITERION = 'friedman_mse', DT_MIN_INPURITY_DECREASE = 0.05,  RF_ESTIMATORS = 100, RF_MAX_DEPTH = 10, RF_MIN_SAMPLE_SPLIT = 2,
              RF_CRITERION = 'friedman_mse', RF_MIN_INPURITY_DECREASE = 0.05, XG_ESTIMATORS = 100, XG_LEARNING_RATE = 0.1, XG_MAX_DEPTH = 5, 
              XG_MIN_CHILD_WEIGHT = 1, XG_SUBSAMPLE = 1.0, XG_COL_SAMPLE_BY_TREE = 1.0, XG_GAMMA = 0.0, XG_REG_ALPHA = 0.0, XG_REG_LAMBDA = 1.0, RANDOM_STATE=42):
    evaluator = eval.Eval()
    m = models.Models()

    partitionDatasets = trainTestSplitter.partitionData(df, NUM_PARTITIONS)
    linRegResults, polyRegResults, ridgeRegResults, lassoRegResults, elasticRegResults, knnRegResults, dtRegResults, rfRegResults, XgRegResults, pcrRegResults = [], [], [], [], [], [], [], [], [], []
    
    for partition in range(5):
        training = partitionDatasets.copy()
        test = training.pop(partition)
        training = pd.concat(training)
        X_train = training[training.columns[:-1]]
        y_train = training[training.columns[-1]]
        X_test = test[training.columns[:-1]]
        y_test = test[training.columns[-1]]
        #Lin Reg
        linRegModel = m.getAndFitLinReg(X_train, y_train)
        y_pred = m.predictModel(linRegModel, X_test)
        linRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'Linear Regression', 'Partition {}'.format(partition)))
        #Poly Reg
        polyRegModel, polyTransformer = m.getAndFitPolyReg(X_train, y_train, POLY_REG_DEGREE)
        y_pred = m.predictPolyReg(polyRegModel, polyTransformer, X_test)
        polyRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'Polynomial Regression', 'Partition {}, Degree {}'.format(partition, POLY_REG_DEGREE)))
        #Ridge Reg
        ridgeRegModel = m.getAndFitRidgeReg(X_train, y_train, RIDGE_REG_ALPHA)
        y_pred = m.predictModel(ridgeRegModel, X_test)
        ridgeRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'Ridge Regression', 'Partition {}, Alpha {}'.format(partition, RIDGE_REG_ALPHA)))
        #Lasso Reg
        lassoRegModel = m.getAndFitLasso(X_train, y_train, alpha=LASSO_ALPHA, shouldFitIntercept=LASSO_INTERCEPT_FIT)
        y_pred = m.predictModel(lassoRegModel, X_test)
        lassoRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'Lasso Regression', 'Partition {}, Alpha {}, Intercept Fit: {}'.format(partition, LASSO_ALPHA, LASSO_INTERCEPT_FIT)))
        #Elastic Reg
        elasticRegModel = m.getAndFitElasticNet(X_train, y_train, alpha=ELASTIC_ALPHA, l1=ELASTIC_L1, shouldFitIntercept=ELASTIC_SHOULD_FIT, selection=ELASTIC_SELECTION)
        y_pred = m.predictModel(elasticRegModel, X_test)
        elasticRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'Elastic Regression', 'Partition {}, Alpha {}, L1 {}, Intercept Fit {}, selection {}'.format(partition, ELASTIC_ALPHA, ELASTIC_L1, ELASTIC_SHOULD_FIT,ELASTIC_SELECTION )))
        #PCR Reg
        pcrModel = m.getAndFitPCR(X_train, y_train, numComponents=PCR_PARAMETERS, polyDegree=PCR_DEGREE)
        y_pred = m.predictModel(pcrModel, X_test)
        pcrRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'PCR Regression', 'Partition {}, Num Parameters {}, Polynomial Degree {}'.format(partition, PCR_PARAMETERS, PCR_DEGREE)))        
        #KNN
        knnRegModel = m.getAndFitRidgeReg(X_train, y_train, KNN_CLUSTERS)
        y_pred = m.predictModel(knnRegModel, X_test)
        knnRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'KNN Regression', 'Partition {}, Clusters {}'.format(partition, KNN_CLUSTERS)))
        #Decision Tree
        DtRegModel = m.getAndFitDtReg(X_train, y_train, DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, DT_CRITERION, DT_MIN_INPURITY_DECREASE, random_state=42)
        y_pred = m.predictModel(DtRegModel, X_test)
        dtRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'DT Regression', 'Partition {}, Max_Depth: {}, Min_SampleSplit: {},\
                                                 Criterion: {}, Min_impurity_decrease: {}, Random_State: {}'.format(partition, DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, 
                                                DT_CRITERION, DT_MIN_INPURITY_DECREASE, RANDOM_STATE)))
        #Random Forest
        RfRegModel = m.getAndFitForestReg(X_train, y_train, RF_ESTIMATORS, DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, DT_CRITERION, DT_MIN_INPURITY_DECREASE, RANDOM_STATE)
        y_pred = m.predictModel(RfRegModel, X_test)
        rfRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'RF Regression', 'Partition {}, Num_Estimators: {}, Max_Depth: {}, \
                                                Min_SampleSplit: {}, Criterion: {}, Min_impurity_decrease: {}, Random_State: {}'.format(partition, 
                                                RF_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLE_SPLIT, RF_CRITERION, RF_MIN_INPURITY_DECREASE, RANDOM_STATE)))
        
        XgRegModel = m.getAndFitXGReg(X_train, y_train, n_estimators=XG_ESTIMATORS, learning_rate=XG_LEARNING_RATE, max_depth=XG_MAX_DEPTH, 
                                      min_child_weight=XG_MIN_CHILD_WEIGHT, subsample=XG_SUBSAMPLE, colsample_bytree=XG_COL_SAMPLE_BY_TREE, 
                                      gamma=XG_GAMMA, reg_alpha=XG_REG_ALPHA, reg_lambda=XG_REG_LAMBDA)
        y_pred = m.predictModel(XgRegModel, X_test)
        XgRegResults.append(evaluator.THE_WORKS(y_test, y_pred, 'XG Regression', 'Partition {}, Num_Estimators: {}, Learning_Rate: {}, Max_Depth: {}, \
                                                Min_Child_Weight: {}, Subsample: {}, ColSampleByTree: {}, Gamma: {}, Alpha: {}, Lambda: {}'.format(partition, 
                                                XG_ESTIMATORS, XG_LEARNING_RATE, XG_MAX_DEPTH, XG_MIN_CHILD_WEIGHT, XG_SUBSAMPLE, XG_COL_SAMPLE_BY_TREE,
                                                XG_GAMMA, XG_REG_ALPHA, XG_REG_LAMBDA)))
    
    return {'Linear': np.array(linRegResults), 'Poly': np.array(polyRegResults), 'Ridge': np.array(ridgeRegResults), 'Lasso': np.array(lassoRegResults),
            'Elastic': np.array(elasticRegResults), 'PCR': np.array(pcrRegResults), 'KNN': np.array(knnRegResults), 'DT': np.array(dtRegResults), 'RF': np.array(rfRegResults), 'XG': np.array(XgRegResults)}



def runModelsOnTestData(training, test, POLY_REG_DEGREE = 2, RIDGE_REG_ALPHA = 1, LASSO_ALPHA = 1, LASSO_INTERCEPT_FIT=True, 
              ELASTIC_ALPHA=1, ELASTIC_L1=0, ELASTIC_SHOULD_FIT=1, ELASTIC_SELECTION='cyclic', PCR_PARAMETERS = 2, PCR_DEGREE = 3,
              KNN_CLUSTERS = 5, DT_MAX_DEPTH = 10, DT_MIN_SAMPLE_SPLIT = 2,
              DT_CRITERION = 'friedman_mse', DT_MIN_INPURITY_DECREASE = 0.05,  RF_ESTIMATORS = 100, RF_MAX_DEPTH = 10, RF_MIN_SAMPLE_SPLIT = 2,
              RF_CRITERION = 'friedman_mse', RF_MIN_INPURITY_DECREASE = 0.05, XG_ESTIMATORS = 100, XG_LEARNING_RATE = 0.1, XG_MAX_DEPTH = 5, 
              XG_MIN_CHILD_WEIGHT = 1, XG_SUBSAMPLE = 1.0, XG_COL_SAMPLE_BY_TREE = 1.0, XG_GAMMA = 0.0, XG_REG_ALPHA = 0.0, XG_REG_LAMBDA = 1.0):
    evaluator = eval.Eval()
    m = models.Models()

    X_train = training[training.columns[:-1]]
    y_train = training[training.columns[-1]]
    X_test = test[training.columns[:-1]]
    y_test = test[training.columns[-1]]
    #Lin Reg
    linRegModel = m.getAndFitLinReg(X_train, y_train)
    y_pred = m.predictModel(linRegModel, X_test)
    linRegResults = evaluator.THE_WORKS(y_test, y_pred, 'Linear Regression', '')
    #Poly Reg
    polyRegModel, polyTransformer = m.getAndFitPolyReg(X_train, y_train, POLY_REG_DEGREE)
    y_pred = m.predictPolyReg(polyRegModel, polyTransformer, X_test)
    polyRegResults = evaluator.THE_WORKS(y_test, y_pred, 'Polynomial Regression', 'Degree {}'.format(POLY_REG_DEGREE))
    #Ridge Reg
    ridgeRegModel = m.getAndFitRidgeReg(X_train, y_train, RIDGE_REG_ALPHA)
    y_pred = m.predictModel(ridgeRegModel, X_test)
    ridgeRegResults = evaluator.THE_WORKS(y_test, y_pred, 'Ridge Regression', 'Alpha {}'.format(RIDGE_REG_ALPHA))
    #Lasso Reg
    lassoRegModel = m.getAndFitLasso(X_train, y_train, alpha=LASSO_ALPHA, shouldFitIntercept=LASSO_INTERCEPT_FIT)
    y_pred = m.predictModel(lassoRegModel, X_test)
    lassoRegResults = (evaluator.THE_WORKS(y_test, y_pred, 'Lasso Regression', 'Alpha {}, Intercept Fit: {}'.format(LASSO_ALPHA, LASSO_INTERCEPT_FIT)))
    #elastic Reg
    elasticRegModel = m.getAndFitElasticNet(X_train, y_train, alpha=ELASTIC_ALPHA, l1=ELASTIC_L1, shouldFitIntercept=ELASTIC_SHOULD_FIT, selection=ELASTIC_SELECTION)
    y_pred = m.predictModel(elasticRegModel, X_test)
    elasticRegResults = (evaluator.THE_WORKS(y_test, y_pred, 'Elastic Regression', 'Alpha {}, L1 {}, Intercept Fit {}, selection {}'.format(ELASTIC_ALPHA, ELASTIC_L1, ELASTIC_SHOULD_FIT,ELASTIC_SELECTION )))
    #PCR Reg
    pcrModel = m.getAndFitPCR(X_train, y_train, numComponents=PCR_PARAMETERS, polyDegree=PCR_DEGREE)
    y_pred = m.predictModel(pcrModel, X_test)
    pcrRegResults = (evaluator.THE_WORKS(y_test, y_pred, 'PCR Regression','Num Parameters {}, Polynomial Degree {}'.format(PCR_PARAMETERS, PCR_DEGREE)))        
        
    #KNN
    knnRegModel = m.getAndFitRidgeReg(X_train, y_train, KNN_CLUSTERS)
    y_pred = m.predictModel(knnRegModel, X_test)
    knnRegResults = evaluator.THE_WORKS(y_test, y_pred, 'KNN Regression', 'Clusters {}'.format(KNN_CLUSTERS))
    #Decision Tree
    DtRegModel = m.getAndFitDtReg(X_train, y_train, DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, DT_CRITERION, DT_MIN_INPURITY_DECREASE)
    y_pred = m.predictModel(DtRegModel, X_test)
    dtRegResults = evaluator.THE_WORKS(y_test, y_pred, 'DT Regression', 'Max_Depth: {}, Min_SampleSplit: {},\
                                                Criterion: {}, Min_impurity_decrease: {}'.format(DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, 
                                            DT_CRITERION, DT_MIN_INPURITY_DECREASE))
    #Decision Tree
    RfRegModel = m.getAndFitForestReg(X_train, y_train, RF_ESTIMATORS, DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, DT_CRITERION, DT_MIN_INPURITY_DECREASE)
    y_pred = m.predictModel(RfRegModel, X_test)
    rfRegResults = evaluator.THE_WORKS(y_test, y_pred, 'RF Regression', 'Num_Estimators: {}, Max_Depth: {}, \
                                            Min_SampleSplit: {}, Criterion: {}, Min_impurity_decrease: {}'.format( 
                                            RF_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLE_SPLIT, RF_CRITERION, RF_MIN_INPURITY_DECREASE))
    
    XgRegModel = m.getAndFitXGReg(X_train, y_train, n_estimators=XG_ESTIMATORS, learning_rate=XG_LEARNING_RATE, max_depth=XG_MAX_DEPTH, 
                                    min_child_weight=XG_MIN_CHILD_WEIGHT, subsample=XG_SUBSAMPLE, colsample_bytree=XG_COL_SAMPLE_BY_TREE, 
                                    gamma=XG_GAMMA, reg_alpha=XG_REG_ALPHA, reg_lambda=XG_REG_LAMBDA)
    y_pred = m.predictModel(XgRegModel, X_test)
    XgRegResults = evaluator.THE_WORKS(y_test, y_pred, 'XG Regression', 'Num_Estimators: {}, Learning_Rate: {}, Max_Depth: {}, \
                                            Min_Child_Weight: {}, Subsample: {}, ColSampleByTree: {}, Gamma: {}, Alpha: {}, Lambda: {}'.format( 
                                            XG_ESTIMATORS, XG_LEARNING_RATE, XG_MAX_DEPTH, XG_MIN_CHILD_WEIGHT, XG_SUBSAMPLE, XG_COL_SAMPLE_BY_TREE,
                                            XG_GAMMA, XG_REG_ALPHA, XG_REG_LAMBDA))
    return {'Linear': linRegResults, 'Poly': polyRegResults, 'Ridge': ridgeRegResults, 'Lasso': lassoRegResults,
            'Elastic': elasticRegResults, 'PCR': pcrRegResults, 'KNN': knnRegResults, 'DT': dtRegResults, 'RF': rfRegResults, 'XG': XgRegResults}


def parsePartitionResults(results, shouldPrint = False):
    modelResultsDict = {}
    modelPerformanceResultsDict = {}
    for temp in results:
        modelResults = results[temp]
        rmse = []
        percentOfMedian = []
        guesses = []
        gt = []
        r2 = []
        for i in modelResults:
            rmse.append(i['RMSE'])
            percentOfMedian.append(100 * np.absolute(i['RMSE'] / i['GT_Median']))
            guesses.append(i['PRED_Data'])
            gt.append(i['GT_Data'])
            r2.append(i['R2'])
        modelResultsDict[modelResults[0]['Model_Name']] = (gt, guesses)
        if shouldPrint: 
            print(modelResults[0]['Model_Name'])
            print('  RMSE:        {}'.format(np.mean(rmse)))
            print('  RMSE/Median: {}'.format(np.mean(percentOfMedian)))
            print('  R2:          {}'.format(np.mean(r2)))
        modelPerformanceResultsDict[temp] = {'ModelName': modelResults[0]['Model_Name'],
                                             'RMSE': np.mean(rmse), 
                                             'RMSE/Median': np.mean(percentOfMedian),
                                             'R2': np.mean(r2)}
    return modelPerformanceResultsDict

def calculateMeanRMSE(testingDF):
    meanVal = np.mean(testingDF)
    tempArray = np.zeros(len(testingDF)) + meanVal
    print(f'Target Baseline (Mean RMSE): {np.sqrt(np.mean((testingDF - tempArray) ** 2))}')

def computeTrainTestResults(filePath, targetVariable = 'Width', categorical=False, NUM_PARTITIONS=5, POLY_REG_DEGREE = 2, RIDGE_REG_ALPHA = 1, 
              LASSO_ALPHA = 1, LASSO_INTERCEPT_FIT=True, ELASTIC_ALPHA=1, ELASTIC_L1=0, ELASTIC_SHOULD_FIT=1, ELASTIC_SELECTION='cyclic', 
              PCR_PARAMETERS = 2, PCR_DEGREE = 3, KNN_CLUSTERS = 5, DT_MAX_DEPTH = 10, DT_MIN_SAMPLE_SPLIT = 2,
              DT_CRITERION = 'friedman_mse', DT_MIN_INPURITY_DECREASE = 0.05,  RF_ESTIMATORS = 100, RF_MAX_DEPTH = 10, RF_MIN_SAMPLE_SPLIT = 2,
              RF_CRITERION = 'friedman_mse', RF_MIN_INPURITY_DECREASE = 0.05, XG_ESTIMATORS = 100, XG_LEARNING_RATE = 0.1, XG_MAX_DEPTH = 5, 
              XG_MIN_CHILD_WEIGHT = 1, XG_SUBSAMPLE = 1.0, XG_COL_SAMPLE_BY_TREE = 1.0, XG_GAMMA = 0.0, XG_REG_ALPHA = 0.0, XG_REG_LAMBDA = 1.0,
              shouldPrint = False):
    raw_data = trainTestSplitter.collectData(filePath)
    raw_data = raw_data[~raw_data.apply(lambda row: row.astype(str).str.contains('TBD')).any(axis=1)]
    raw_data = raw_data.reset_index(drop=True)
    raw_data = raw_data[raw_data['Width'] != -1]
    #Processing Training Data
    raw_trainingData = trainTestSplitter.removeValidationData(raw_data)
    
    splitTrainingData = trainTestSplitter.splitTargetVariables(raw_trainingData)
    df = splitTrainingData[targetVariable]
    #Processing Test Data
    raw_testingData = trainTestSplitter.removeTrainingData(raw_data)
    splitTestingData = trainTestSplitter.splitTargetVariables(raw_testingData)
    testingDF = splitTestingData[targetVariable]

    if categorical == False:
        df = removeUnwantedCols(splitTrainingData[targetVariable])
        testingDF = removeUnwantedCols(splitTestingData[targetVariable])
    elif categorical == True: 
        df = removeUnwantedContCols(splitTrainingData[targetVariable])
        testingDF = removeUnwantedContCols(splitTestingData[targetVariable])
    else:
        print("ERROR: UNKNOWN FEATURE SELECTION TYPE")
    print('Training Data: ', df.shape)
    print('Testing Data: ', testingDF.shape)
    calculateMeanRMSE(testingDF[targetVariable])

    results = runModels(df, NUM_PARTITIONS = NUM_PARTITIONS, POLY_REG_DEGREE = POLY_REG_DEGREE, RIDGE_REG_ALPHA = RIDGE_REG_ALPHA,
              LASSO_ALPHA = LASSO_ALPHA, LASSO_INTERCEPT_FIT=LASSO_INTERCEPT_FIT,ELASTIC_ALPHA=ELASTIC_ALPHA, ELASTIC_L1=ELASTIC_L1, ELASTIC_SHOULD_FIT=ELASTIC_SHOULD_FIT, ELASTIC_SELECTION=ELASTIC_SELECTION, 
              PCR_PARAMETERS=PCR_PARAMETERS, PCR_DEGREE=PCR_DEGREE, KNN_CLUSTERS = KNN_CLUSTERS, DT_MAX_DEPTH = DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT = DT_MIN_SAMPLE_SPLIT,
              DT_CRITERION = DT_CRITERION, DT_MIN_INPURITY_DECREASE = DT_MIN_INPURITY_DECREASE,  RF_ESTIMATORS = RF_ESTIMATORS,
              RF_MAX_DEPTH = RF_MAX_DEPTH, RF_MIN_SAMPLE_SPLIT = RF_MIN_SAMPLE_SPLIT, RF_CRITERION = RF_CRITERION,
              RF_MIN_INPURITY_DECREASE = RF_MIN_INPURITY_DECREASE, XG_ESTIMATORS = XG_ESTIMATORS, XG_LEARNING_RATE = XG_LEARNING_RATE, 
              XG_MAX_DEPTH = XG_MAX_DEPTH, XG_MIN_CHILD_WEIGHT = XG_MIN_CHILD_WEIGHT, XG_SUBSAMPLE = XG_SUBSAMPLE,
              XG_COL_SAMPLE_BY_TREE = XG_COL_SAMPLE_BY_TREE, XG_GAMMA = XG_GAMMA, XG_REG_ALPHA = XG_REG_ALPHA, XG_REG_LAMBDA = XG_REG_LAMBDA)
    
    parsedResults = parsePartitionResults(results) 


    testResults = runModelsOnTestData(df, testingDF, POLY_REG_DEGREE = POLY_REG_DEGREE, RIDGE_REG_ALPHA = RIDGE_REG_ALPHA,
              LASSO_ALPHA = LASSO_ALPHA, LASSO_INTERCEPT_FIT=LASSO_INTERCEPT_FIT,ELASTIC_ALPHA=ELASTIC_ALPHA, ELASTIC_L1=ELASTIC_L1, ELASTIC_SHOULD_FIT=ELASTIC_SHOULD_FIT, ELASTIC_SELECTION=ELASTIC_SELECTION, 
              PCR_PARAMETERS=PCR_PARAMETERS, PCR_DEGREE=PCR_DEGREE, KNN_CLUSTERS = KNN_CLUSTERS, DT_MAX_DEPTH = DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT = DT_MIN_SAMPLE_SPLIT,
              DT_CRITERION = DT_CRITERION, DT_MIN_INPURITY_DECREASE = DT_MIN_INPURITY_DECREASE,  RF_ESTIMATORS = RF_ESTIMATORS,
              RF_MAX_DEPTH = RF_MAX_DEPTH, RF_MIN_SAMPLE_SPLIT = RF_MIN_SAMPLE_SPLIT, RF_CRITERION = RF_CRITERION,
              RF_MIN_INPURITY_DECREASE = RF_MIN_INPURITY_DECREASE, XG_ESTIMATORS = XG_ESTIMATORS, XG_LEARNING_RATE = XG_LEARNING_RATE, 
              XG_MAX_DEPTH = XG_MAX_DEPTH, XG_MIN_CHILD_WEIGHT = XG_MIN_CHILD_WEIGHT, XG_SUBSAMPLE = XG_SUBSAMPLE,
              XG_COL_SAMPLE_BY_TREE = XG_COL_SAMPLE_BY_TREE, XG_GAMMA = XG_GAMMA, XG_REG_ALPHA = XG_REG_ALPHA, XG_REG_LAMBDA = XG_REG_LAMBDA)

    difference = {}
    for i in testResults:
        difference[i] = {'Model': '{}'.format(parsedResults[i]['ModelName']),
                         'Testing RMSE':  '{:.08}'.format(testResults[i]['RMSE']),
                         'Change in RMSE': '{:.08}'.format(parsedResults[i]['RMSE'] - testResults[i]['RMSE']), 
                         'Testing R2':  '{:.08}'.format(testResults[i]['R2']), 
                         'Change in R2':   '{:.08}'.format(parsedResults[i]['R2'] - testResults[i]['R2'])}
        if shouldPrint:
            print(difference[i])
        


if __name__ == '__main__':
    df = pd.read_excel('Solar_cell_Devices.xlsx')
    print('With Tuning')
    results = runModels(df)
    parsedResults = parsePartitionResults(results, True) 
    # computeTrainTestResults('../training/Stage1/STAGE_1_Data-WH.xlsx', categorical=False, targetVariable='Width', shouldPrint=True)
    # computeTrainTestResults('../training/Stage1/STAGE_1_Data-WH.xlsx', categorical=True, targetVariable='Width', shouldPrint=True)