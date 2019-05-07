# -*- coding: utf-8 -*-
"""
@author: Shreyas Menon
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import cleaning 


if __name__ == '__main__':
    

    df = pd.read_csv('knowledge_base.csv',index_col=0)
    
    df = df.reset_index(drop=True)
            
    df['query_request'] = cleaning.tokenize_lemmatize(list(df.query_request), stop_words = True, extract_entities = False)
    
        
    df['intent'] = df.intent.apply(lambda x : str(x).lower())
    
    #use this for optimization
    #df['query_request'] = df['query_request'].apply(lambda x: np.nan if x in list_null_values else x)
    
    #use this for optimization
    #df.dropna(inplace = True)
    
    X_train, X_test, y_train, y_test = train_test_split(df['query_request'], df['intent'], test_size=0.2, random_state=0)
    
    parameters = {'clf__C':[1,10,100,1000],'clf__gamma':[1,0.1,0.001,0.0001], 'clf__kernel':['linear','rbf'] , 'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)]}

    mdl = Pipeline([('vectorizer', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SVC())])
    
    # the metric used to select the best parameters
    metric =  "accuracy"

    # GridSearch also uses cross validation
    gs_pipeline = GridSearchCV(mdl, param_grid=parameters, refit = True , verbose = 2, scoring=metric , cv=5 )

    gs_model = gs_pipeline.fit(X_train, y_train)
    
    for param_name in gs_model.best_params_:
        print(param_name,": ",gs_model.best_params_[param_name])
        

    print("best f1 score:", gs_model.best_score_)
    
    predict_classification = gs_model.predict(X_test)
    
    print(classification_report(y_test,predict_classification))
    
    
    svc_model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))) , ('tfidf', TfidfTransformer(use_idf=True)) ,('clf', SVC(kernel='rbf', C=10, gamma=0.1, verbose = True)) ])

    svc_model.fit(X_train, y_train)
    
    
    #predicted= svc_model.predict(X_test)
    
    #print(classification_report(y_test,predicted))
    
    '''
    #Uncomment to write the classifier into a pickel file
    with open('classifier.pkl', 'wb') as fout:
        cPickle.dump(svc_model, fout)
    '''
    
        