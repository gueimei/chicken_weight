# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 23:31:13 2019

@author: may
"""

from sklearn.svm import SVR
#from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from pre_train import load_xml
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error#, mean_absolute_error, r2_score
from sklearn.externals import joblib

if __name__ == "__main__":
    
    ac_data, train_data = load_xml()
    ac_data = np.array(ac_data)
    train_data = np.array(train_data)
    x_train, x_test, y_train, y_test = train_test_split(train_data, ac_data, test_size=0.1, random_state=33)

    
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-1, 1e-2, ],
                     'C': [1, 10, 100, 1000, 1e0, 1e1, 1e2, 1e3]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    linear_svr = GridSearchCV(SVR(), tuned_parameters, cv=5)

    # 訓練
    #y_train = np.ravel(y_train)
    linear_svr.fit(x_train, y_train)

    joblib.dump(linear_svr,'svr.pkl')
    # 預測 保存預測結果
    linear_svr_y_predict = linear_svr.predict(x_test)
    print(linear_svr.best_params_)
    
    print("linear函數支持向量機的均方誤差為:", mean_squared_error(y_test, linear_svr_y_predict))
    print(y_test)
    print(linear_svr_y_predict)
    
