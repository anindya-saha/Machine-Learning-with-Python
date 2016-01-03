
# coding: utf-8

# In[59]:

import pandas as pd
import numpy as np
import xgboost as xgb


# In[61]:
if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    
    # In[62]:
    
    # drop unnecessary columns, these columns won't be useful in analysis and prediction
    test_df.drop(['QuoteNumber'], axis=1, inplace=True)
    train_df.drop(['QuoteNumber'], axis=1, inplace=True)
    
    
    # In[63]:
    
    train_df.info()
    
    
    # In[64]:
    
    test_df.info()
    
    
    # In[65]:
    
    train_df.head(3)
    
    
    # In[66]:
    
    # Convert Date to Year, Month, and Week
    train_df['Date'] = pd.to_datetime(pd.Series(train_df['Original_Quote_Date']))
    train_df['Year']  = train_df['Date'].apply(lambda x: int(str(x)[:4]))
    train_df['Month'] = train_df['Date'].apply(lambda x: int(str(x)[5:7]))
    train_df['Weekday']  = train_df['Date'].dt.dayofweek
    
    test_df['Date'] = pd.to_datetime(pd.Series(test_df['Original_Quote_Date']))
    test_df['Year']  = test_df['Date'].apply(lambda x: int(str(x)[:4]))
    test_df['Month'] = test_df['Date'].apply(lambda x: int(str(x)[5:7]))
    test_df['Weekday']  = test_df['Date'].dt.dayofweek
    
    train_df.drop(['Original_Quote_Date', 'Date'], axis=1, inplace=True)
    test_df.drop(['Original_Quote_Date', 'Date'], axis=1, inplace=True)
    
    
    # In[67]:
    
   
    # In[76]:
    
    # There are some columns with non-numerical values(i.e. dtype='object'),
    # So, We will create a corresponding unique numerical value for each non-numerical value in a column of training and testing set.
    
    from sklearn import preprocessing
    
    for f in train_df.columns:
        if train_df[f].dtype=='object':
            print(f)
            lbl_encoder = preprocessing.LabelEncoder()
            lbl_encoder.fit(np.unique(list(train_df[f].values) + list(test_df[f].values)))
            train_df[f] = lbl_encoder.transform(list(train_df[f].values))
            test_df[f] = lbl_encoder.transform(list(test_df[f].values))
    
    
    # In[77]:
    
    # define training and testing sets
    y_train = train_df['QuoteConversion_Flag']
    X_train = train_df.drop('QuoteConversion_Flag', axis=1)
    X_test  = test_df.copy()
    X_test = X_test[X_train.columns.tolist()] # maintain same column order between train and test data
    
    
    # In[78]:
    
    print(len(X_train.columns))
    print(len(X_test.columns))
    
    
    # In[91]:
    
    # Xgboost 
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic',
                            nthread=-1,
                            silent=True)
    
    
    # In[98]:
    
    from sklearn.grid_search import GridSearchCV
    
    param_grid = {'max_depth': [2,4,6,8,10],
                  'n_estimators': [50,100,200,500,1000],
                  'learning_rate': [0.1, 0.05, 0.02, 0.01],
                  'subsample': [0.9, 1.0],
                  'colsample_bytree': [0.8, 1.0]}
    
    gs = GridSearchCV(xgb_clf,
                      param_grid,
                      scoring='roc_auc',
                      cv=5,
                      n_jobs=-1,
                      verbose=1)
    
    gs.fit(X_train, y_train)
    gs.best_score_, gs.best_params_
    
    
    # In[ ]:
    
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    
    
    # In[ ]:
    
    # Create submission
    sample = pd.read_csv('data/sample_submission.csv')
    sample.QuoteConversion_Flag = y_pred_proba
    sample.head(10)
    
    
    # In[ ]:
    
    sample.to_csv('xgb_benchmark.csv', index=False)

