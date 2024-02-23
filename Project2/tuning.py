import numpy as np
import pandas as pd
import models
import trainTestSplitter, eval, models
import random
import warnings
warnings.filterwarnings("ignore")


NUM_PARTITIONS = 5
evaluator = eval.Eval()
m = models.Models()
'''
Complete list of tuned parameters
Degree - Polyreg
Alpha - Ridge
Neighbors - KNN
max_depth, min_samples_split, criterion, min_impurity_decrease - DT, RF
Num estimators - RF, XG
n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda - XG
'''
def isList(inputParam):
    return isinstance(inputParam, list)

def constructEmptyParamList():
    return {'Degree': None, 'Alpha': None, 'Neighbors': None, 'Max_Depth': None, 'Min_Samples_Split': None, 
            'Criterion': None, 'Min_Impurity_Decrease': None, 'Num_Estimators': None, 'Learning_Rate': None, 
            'Min_Child_Weight':None, 'Subsample':None, 'Colsample_Bytree': None, 'Gamma': None, 
            'Reg_Alpha': None, 'Reg_Lambda':None}

class Tuning:
    def __init__(self):
        pass

    def tunePolyReg(self, data, degree = [1, 2, 3, 4, 5]):
        params = {}
        for i in degree:
            models.getAndFitPolyReg(self.X, self.y, i)
            params['Degree - {}'.format(i)]
    def tunePoly(self, df, POLY_DEGREE):
        print('Beginning Polynomial Tuning')
        idList = {}
        idRawScore = {}
        idScores = {}
        counter = 0
        for d in POLY_DEGREE:
            idList[counter] = [d]
            idRawScore[counter] = []
            counter += 1
        
        print('\tNumber of Parameter Combinations: {}'.format(len(idList)))
        for combination in idList:
            #print('ID: {}, Combinations: {}'.format(combination, idList[combination]))
            currentDegree = idList[combination][0]
            partitionDatasets = trainTestSplitter.partitionData(df, NUM_PARTITIONS)
            for partition in range(5):
                training = partitionDatasets.copy()
                test = training.pop(partition)
                training = pd.concat(training)
                X_train = training[training.columns[:-1]]
                y_train = training[training.columns[-1]]
                X_test = test[training.columns[:-1]]
                y_test = test[training.columns[-1]]
                np.random.seed = 42
                random.seed = 42
                polyRegModel, polyTransformer = m.getAndFitPolyReg(X_train, y_train, currentDegree)
                y_pred = m.predictPolyReg(polyRegModel, polyTransformer, X_test)
                idRawScore[combination].append(evaluator.THE_WORKS(y_test, y_pred, 'Poly Regression', 'Partition {}, Degree: {}'.format(partition, currentDegree)))
        print('\tCompleted Parameter Evaluation')
        for result in idRawScore:
            scores = []
            for i in idRawScore[result]:
                scores.append(i['RMSE'])
            idScores[result] = np.mean(scores)
            #print(idList[result], np.mean(scores))
        # Find the model ID with the lowest loss score
        best_model_id = min(idScores, key=idScores.get)
        best_loss = idScores[best_model_id]

        print(f"\tThe model with the lowest loss is {best_model_id} with a loss of {best_loss}")
        print('\t', idList[best_model_id])

    def tuneDT(self, df, DT_MAX_DEPTH, DT_MIN_SAMPLE_SPLIT, DT_CRITERION, DT_MIN_INPURITY_DECREASE):
        print('Beginning Decision Tree Tuning')
        idList = {}
        idRawScore = {}
        idScores = {}
        counter = 0
        for depth in DT_MAX_DEPTH:
            for sampleSplit in DT_MIN_SAMPLE_SPLIT:
                for criterion in DT_CRITERION:
                    for impurity in DT_MIN_INPURITY_DECREASE:
                        # print('Depth: {}, SampleSplit: {}, Criterion: {}, ImpurityDecrease: {}'.format(depth, sampleSplit, criterion, impurity))
                        idList[counter] = [depth, sampleSplit, criterion, impurity]
                        idRawScore[counter] = []
                        counter += 1
        
        print('\tNumber of Parameter Combinations: {}'.format(len(idList)))
        for combination in idList:
            #print('ID: {}, Combinations: {}'.format(combination, idList[combination]))
            currentMaxDepth, currentMinSampleSplit, currentCriterion, currentMinImpurityDecrease = idList[combination]
            partitionDatasets = trainTestSplitter.partitionData(df, NUM_PARTITIONS)
            for partition in range(5):
                training = partitionDatasets.copy()
                test = training.pop(partition)
                training = pd.concat(training)
                X_train = training[training.columns[:-1]]
                y_train = training[training.columns[-1]]
                X_test = test[training.columns[:-1]]
                y_test = test[training.columns[-1]]
                np.random.seed = 42
                random.seed = 42
                DtRegModel = m.getAndFitDtReg(X_train, y_train, currentMaxDepth, currentMinSampleSplit, currentCriterion, currentMinImpurityDecrease)
                y_pred = m.predictModel(DtRegModel, X_test)
                idRawScore[combination].append(evaluator.THE_WORKS(y_test, y_pred, 'DT Regression', 'Partition {}, Max_Depth: {}, Min_SampleSplit: {},\
                                                        Criterion: {}, Min_impurity_decrease: {}'.format(partition, currentMaxDepth, currentMinSampleSplit,
                                                        currentCriterion, currentMinImpurityDecrease)))
        print('\tCompleted Parameter Evaluation')
        for result in idRawScore:
            scores = []
            for i in idRawScore[result]:
                scores.append(i['RMSE'])
            idScores[result] = np.mean(scores)
            #print(idList[result], np.mean(scores))
        # Find the model ID with the lowest loss score
        best_model_id = min(idScores, key=idScores.get)
        best_loss = idScores[best_model_id]

        print(f"\tThe model with the lowest loss is {best_model_id} with a loss of {best_loss}")
        print('\t', idList[best_model_id])

    def tuneRF(self, df, RF_ESTIMATORS,DT_MAX_DEPTH , DT_MIN_SAMPLE_SPLIT, DT_CRITERION, DT_MIN_INPURITY_DECREASE):
        idList = {}
        idRawScore = {}
        idScores = {}
        counter = 0
        for estimators in RF_ESTIMATORS:
            for depth in DT_MAX_DEPTH:
                for sampleSplit in DT_MIN_SAMPLE_SPLIT:
                    for criterion in DT_CRITERION:
                        for impurity in DT_MIN_INPURITY_DECREASE:
                            # print('Depth: {}, SampleSplit: {}, Criterion: {}, ImpurityDecrease: {}'.format(depth, sampleSplit, criterion, impurity))
                            idList[counter] = [estimators, depth, sampleSplit, criterion, impurity]
                            idRawScore[counter] = []
                            counter += 1
        print('Beginning Random Forest')
        print('\tNumber of Parameter Combinations: {}'.format(len(idList)))
        for combination in idList:
            #print('ID: {}, Combinations: {}'.format(combination, idList[combination]))
            currentEstimators, currentMaxDepth, currentMinSampleSplit, currentCriterion, currentMinImpurityDecrease = idList[combination]
            partitionDatasets = trainTestSplitter.partitionData(df, NUM_PARTITIONS)
            for partition in range(5):
                training = partitionDatasets.copy()
                test = training.pop(partition)
                training = pd.concat(training)
                X_train = training[training.columns[:-1]]
                y_train = training[training.columns[-1]]
                X_test = test[training.columns[:-1]]
                y_test = test[training.columns[-1]]
                np.random.seed = 42
                random.seed = 42
                RfRegModel = m.getAndFitForestReg(X_train, y_train, currentEstimators, currentMaxDepth, currentMinSampleSplit, currentCriterion, currentMinImpurityDecrease, randomState=42)
                y_pred = m.predictModel(RfRegModel, X_test)
                idRawScore[combination].append(evaluator.THE_WORKS(y_test, y_pred, 'RF Regression', 'Partition {}, Num_Estimators: {}, Max_Depth: {}, \
                                                Min_SampleSplit: {}, Criterion: {}, Min_impurity_decrease: {}'.format(partition, 
                                                currentEstimators, currentMaxDepth, currentMinSampleSplit, currentCriterion, currentMinImpurityDecrease)))
        print('\tCompleted Parameter Evaluation')
        for result in idRawScore:
            scores = []
            for i in idRawScore[result]:
                scores.append(i['RMSE'])
            idScores[result] = np.mean(scores)
            #print(idList[result], np.mean(scores))
        # Find the model ID with the lowest loss score
        best_model_id = min(idScores, key=idScores.get)
        best_loss = idScores[best_model_id]

        print(f"\tThe model with the lowest loss is {best_model_id} with a loss of {best_loss}")
        print('\t', idList[best_model_id])

    def tuneXG(self, df, XG_ESTIMATORS, XG_LEARNING_RATE, XG_MAX_DEPTH, XG_MIN_CHILD_WEIGHT, XG_SUBSAMPLE, XG_COL_SAMPLE_BY_TREE, XG_GAMMA, XG_REG_ALPHA, XG_REG_LAMBDA):
        idList = {}
        idRawScore = {}
        idScores = {}
        counter = 0
        for estimators in XG_ESTIMATORS:
            for lr in XG_LEARNING_RATE:
                for depth in XG_MAX_DEPTH:
                    for childWeight in XG_MIN_CHILD_WEIGHT:
                        for subSample in XG_SUBSAMPLE:
                            for colSample in XG_COL_SAMPLE_BY_TREE:
                                for gamma in XG_GAMMA:
                                    for alpha in XG_REG_ALPHA:
                                        for lamb in XG_REG_LAMBDA:
                                            # print('Depth: {}, SampleSplit: {}, Criterion: {}, ImpurityDecrease: {}'.format(depth, sampleSplit, criterion, impurity))
                                            idList[counter] = [estimators, lr, depth, childWeight, subSample, colSample, gamma, alpha, lamb]
                                            idRawScore[counter] = []
                                            counter += 1
        print('Beginning XGBoost')
        print('\tNumber of Parameter Combinations: {}'.format(len(idList)))
        for combination in idList:
            print('ID: {}'.format(combination))#, Combinations: {}'.format(combination, idList[combination]))
            currentEstimators, currentLearningRate, currentDepth, currentMinChildWeight, currentSample, currentColSample, currentGamma, currentAlpha, currentLambda = idList[combination]
            partitionDatasets = trainTestSplitter.partitionData(df, NUM_PARTITIONS)
            for partition in range(5):
                training = partitionDatasets.copy()
                test = training.pop(partition)
                training = pd.concat(training)
                X_train = training[training.columns[:-1]]
                y_train = training[training.columns[-1]]
                X_test = test[training.columns[:-1]]
                y_test = test[training.columns[-1]]
                np.random.seed = 42
                random.seed = 42
            
                XgRegModel = m.getAndFitXGReg(X_train, y_train, n_estimators=currentEstimators, learning_rate=currentLearningRate, max_depth=currentDepth, 
                                      min_child_weight=currentMinChildWeight, subsample=currentSample, colsample_bytree=currentColSample, 
                                      gamma=currentGamma, reg_alpha=currentAlpha, reg_lambda=currentLambda)
                y_pred = m.predictXgModel(XgRegModel, X_test)
                idRawScore[combination].append(evaluator.THE_WORKS(y_test, y_pred, 'XG Regression', 'Partition {}, Num_Estimators: {}, Learning_Rate: {}, Max_Depth: {}, \
                                                        Min_Child_Weight: {}, Subsample: {}, ColSampleByTree: {}, Gamma: {}, Alpha: {}, Lambda: {}'.format(partition, 
                                                        currentEstimators, currentLearningRate, currentDepth, currentMinChildWeight, currentSample, currentColSample,
                                                        currentGamma, currentAlpha, currentLambda)))
        print('\tCompleted Parameter Evaluation')
        for result in idRawScore:
            scores = []
            for i in idRawScore[result]:
                scores.append(i['RMSE'])
            idScores[result] = np.mean(scores)
            #print(idList[result], np.mean(scores))
        # Find the model ID with the lowest loss score
        best_model_id = min(idScores, key=idScores.get)
        best_loss = idScores[best_model_id]

        print(f"\tThe model with the lowest loss is {best_model_id} with a loss of {best_loss}")
        print('\t', idList[best_model_id])
        
        

if __name__ == '__main__':
    t = Tuning()
    df = pd.read_excel('Solar_cell_Devices.xlsx')
    data = df.drop(columns=['Temperature'])
    print('For Width')
    data = data.reset_index(drop=True)
    t.tunePoly(df = data, POLY_DEGREE= [2, 3, 4, 5,])
    t.tuneDT(df = data, DT_MAX_DEPTH = [2, 5, 10, 15, 20], DT_MIN_SAMPLE_SPLIT = [2, 5, 10, 20, 25, 50], 
                 DT_CRITERION = ['friedman_mse', 'squared_error'], DT_MIN_INPURITY_DECREASE = [0.0, 0.01, 0.05, 0.1])
    t.tuneRF(df = data, RF_ESTIMATORS= [2, 5, 10, 50, 75, 100, 125], DT_MAX_DEPTH = [2, 5, 10, 15, 20], DT_MIN_SAMPLE_SPLIT = [2, 5, 10, 20, 25, 50], 
                 DT_CRITERION = ['friedman_mse', 'squared_error'], DT_MIN_INPURITY_DECREASE = [0.0, 0.01, 0.05, 0.1])
