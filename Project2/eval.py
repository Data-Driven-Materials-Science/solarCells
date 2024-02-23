# Requirements
import numpy as np
from sklearn.metrics import r2_score


class Eval:
    '''
    Constructor Input:
        - [String] Name:            Name of Model Used to Generate Results
        - [1D Numpy Array] GT:      Correct Values
        - [1D Numpy Array] Pred:    Model Predicted Values
    '''
    def __init__(self):
        pass
    """
    Calculate Mean Squared Error (MSE).
    """
    def MSE(self, GT, PRED):
        return np.mean((GT - PRED) ** 2)
    """
    Calculate Root Mean Squared Error (RMSE).
    """
    def RMSE(self, GT, PRED):
        return np.sqrt(self.MSE(GT, PRED))
    """
    Calculate Mean Absolute Error (MAE).
    """
    def MAE(self, GT, PRED):
        return np.mean(np.abs(GT - PRED))
    """
    Calculate Mean Percentage Error (MPE).
    """
    def MPE(self, GT, PRED):
        GT[GT == 0] = 0.01
        return np.mean((GT - PRED) / GT) * 100
    
    def R2(self, GT, PRED):
        return r2_score(GT, PRED)
    
    def STD(self, data):
        return np.std(data)
    
    def MEAN(self, data):
        return np.mean(data)
    
    def MEDIAN(self, data):
        return np.median(data)
    
    def RANGE(self, data):
        return max(data) - min(data)
    
    def MinERROR(self, GT, PRED):
        return min(np.abs(GT - PRED))
    
    def MaxERROR(self, GT, PRED):
        return max(np.abs(GT - PRED))

    def THE_WORKS(self, GT, PRED, Name, Notes):
        outputDict = {'GT_Mean': self.MEAN(GT), 'PRED_Mean': self.MEAN(PRED), 
                      'GT_Median': self.MEDIAN(GT), 'PRED_Median': self.MEDIAN(PRED),
                      'GT_Range': self.RANGE(GT), 'PRED_Range': self.RANGE(PRED),
                      'GT_STD': self.STD(GT), 'PRED_STD': self.STD(PRED), 
                      'MSE': self.MSE(GT, PRED), 'RMSE': self.RMSE(GT, PRED), 'MAE': self.MAE(GT, PRED), 'MPE': self.MPE(GT, PRED),
                      'MinError': self.MinERROR(GT, PRED), 'MaxError': self.MaxERROR(GT, PRED), 'R2': self.R2(GT, PRED),
                      'GT_Data': GT, 'PRED_Data': PRED, 
                      'Model_Name': Name, 'Model_notes': Notes}
        return outputDict
    
    def printTheWorks(self, outputDict):
        print('----------------------------------------')
        print('Overview:')
        print('--------------------')
        print('Model Name:              {}'.format(outputDict['Model_Name']))
        print('Number of Datapoints:    {}'.format(len(outputDict['GT_Data'])))
        print('Model Parameters:        {}'.format(outputDict['Model_notes']))
        print('--------------------')
        print('General Statistics:')
        print('--------------------')
        print('GT Mean:   {:.03}  \t Pred Mean:   {:.03}'.format(outputDict['GT_Mean'], outputDict['PRED_Mean']))
        print('GT Median: {:.03}  \t Pred Median: {:.03}'.format(outputDict['GT_Median'], outputDict['PRED_Median']))
        print('GT Range:  {:.03}  \t Pred Range:  {:.03}'.format(outputDict['GT_Range'], outputDict['PRED_Range']))
        print('GT STD:    {:.03}  \t Pred STD:    {:.03}'.format(outputDict['GT_STD'], outputDict['PRED_STD']))
        print('--------------------')
        print('Performance Metrics')
        print('--------------------')
        print('Min Error: {:.03}  \t Max Error:   {:.03}'.format(outputDict['MinError'], outputDict['MaxError']))
        print('R2: {:.03}'.format(outputDict['R2']))
        print('MSE:  {:.03}'.format(outputDict['MSE']))
        print('RMSE: {:.03}'.format(outputDict['RMSE']))
        print('MAE:  {:.03}'.format(outputDict['MAE']))
        print('MPE:  {:.03}%'.format(outputDict['MPE']))
        print('----------------------------------------')

# Example usage:
if __name__ == "__main__":
    # Create an instance of Eval
    evaluator = Eval()
    scores = evaluator.THE_WORKS(GT=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6, 7]), PRED=np.array([0.9, 2.1, 2.8, 3.7, 4.6, 6, 7]), Name="Polynomial Regression", Notes='Degree=3')
    evaluator.printTheWorks(scores)






