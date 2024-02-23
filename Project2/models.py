import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet




class Models:

    def __init__(self):
        pass

    def predictModel(self, model, X_test):
        return model.predict(X_test)
    '''
    Linear Regression Models
    '''
    def getAndFitLinReg(self, X, y):
        return LinearRegression().fit(X, y)

    '''
    Polynomial Regression Models
    '''
    def getAndFitPolyReg(self, X, y, degree):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        return model.fit(X_poly, y), poly
    
    def predictPolyReg(self, model, poly, X_test):
        X_test_poly = poly.transform(X_test)
        return model.predict(X_test_poly)

    '''
    Ridge Regression Model
    '''
    def getAndFitRidgeReg(self, X, y, alpha):
        ridge_model = Ridge(alpha=alpha)
        return ridge_model.fit(X, y)
    
    '''
    KNN Regression Model
    '''
    def getAndFitKNNReg(self, X, y, neighbors):
        knn = KNeighborsRegressor(n_neighbors=neighbors)
        return knn.fit(X, y)

    '''
    DT Regression Model
    '''
    def getAndFitDtReg(self, X, y, max_depth, min_samples_split, criterion, min_impurity_decrease, random_state=None):
        if random_state == None:
            tree_model = DecisionTreeRegressor(max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        criterion=criterion,
                                        min_impurity_decrease=min_impurity_decrease)
        else:
            tree_model = DecisionTreeRegressor(max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        criterion=criterion,
                                        min_impurity_decrease=min_impurity_decrease, 
                                        random_state=random_state)
        return tree_model.fit(X, y)

    
    '''
    Random Forest Regression Model
    '''
    def getAndFitForestReg(self, X, y, num_estimators, max_depth, min_samples_split, criterion, min_impurity_decrease, randomState = None):
        if randomState == None:
            forest_model = RandomForestRegressor(n_estimators=num_estimators, 
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            criterion=criterion,
                                            min_impurity_decrease=min_impurity_decrease)
        else:
            forest_model = RandomForestRegressor(n_estimators=num_estimators, 
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            criterion=criterion,
                                            min_impurity_decrease=min_impurity_decrease, random_state=randomState)
        return forest_model.fit(X, y)

    
    '''
    XgBoost Regression Model
    '''
    def getAndFitXGReg(self, X, y, n_estimators=100, learning_rate=0.1, max_depth=5, min_child_weight=1, subsample=1.0, 
                                    colsample_bytree=1.0, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0):
        xgb_model = XGBRegressor(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         gamma=gamma,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         random_state=42)
        return xgb_model.fit(X, y)
    

    '''ADD SVR'''

    def getAndFitSVR(self, X, y):
        pass


    '''PCR Model'''
    def getAndFitPCR(self, X, y, numComponents = 5, polyDegree = 2):
        pipeline = Pipeline([('scaling', StandardScaler()),  # Standardize features
                            ('pca', PCA(n_components=numComponents)),  # Apply PCA
                            ('polynomial', PolynomialFeatures(degree=polyDegree)),  # Apply Polynomial features
                            ('regression', LinearRegression())  # Apply Polynomial Regression
                        ])
        pipeline.fit(X, y)
        return pipeline


    '''Lasso Model'''
    def getAndFitLasso(self, X, y, alpha = 1, shouldFitIntercept = False):
        model = Lasso(alpha=alpha, fit_intercept=shouldFitIntercept)
        model.fit(X, y)
        return model

    '''Elastic Model'''
    def getAndFitElasticNet(self, X, y, alpha = 1, l1 = 0, shouldFitIntercept = False, selection='cyclic'):
        model = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=shouldFitIntercept, selection=selection)
        model.fit(X, y)
        return model


# Example usage:
if __name__ == "__main__":
    m = Models()
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X_test = np.array([[3, 5], [4, 6], [1, 3]])
    linRegModel = m.getAndFitLinReg(X, y)
    newLinRegResults = m.predictModel(linRegModel, X_test)
    print(np.dot(np.array([[3, 5], [4, 6], [1,3]]), np.array([1,2])) + 3)
    print('Linear', newLinRegResults)

    polyRegModel, polyTransformer = m.getAndFitPolyReg(X, y, 3)
    newPolyRegResults = m.predictPolyReg(polyRegModel, polyTransformer, X_test)
    print('Poly', newPolyRegResults)

    ridgeRegModel = m.getAndFitRidgeReg(X, y, 1)
    newRidgeRegResults = m.predictModel(ridgeRegModel, X_test)
    print('Ridge', newRidgeRegResults)

    lassoRegModel = m.getAndFitLasso(X, y, 1, True)
    newLassoRegResults = m.predictModel(lassoRegModel, X_test)
    print('Lasso', newLassoRegResults)

    # elasticRegModel = m.getAndFitElasticNet(X, y, 1, 0, True, 'cyclic')
    # newElasticRegResults = m.predictModel(elasticRegModel, X_test)
    # print('Elastic', newElasticRegResults)

    pcrModel = m.getAndFitPCR(X, y, numComponents=1, polyDegree=2)
    pcrResults = m.predictModel(pcrModel, X_test)
    print('PCR', pcrResults)

    knnRegModel = m.getAndFitKNNReg(X, y, 2)
    knnRegResults = m.predictModel(knnRegModel, X_test)
    print('KNN', knnRegResults)

    DtRegModel = m.getAndFitDtReg(X, y, 5, 2, 'friedman_mse', 0)
    DtRegResults = m.predictModel(DtRegModel, X_test)
    print('DT', DtRegResults)

    ForestRegModel = m.getAndFitForestReg(X, y, 30, 5, 2, 'friedman_mse', 0)
    ForestRegResults = m.predictModel(ForestRegModel, X_test)
    print('Forest', ForestRegResults)

    XgRegModel = m.getAndFitXGReg(X, y, n_estimators=100, learning_rate=0.1, max_depth=5, min_child_weight=1, subsample=1.0, 
                                    colsample_bytree=1.0, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0)
    XgRegResults = m.predictModel(XgRegModel, X_test)
    print('XG', XgRegResults)

    






