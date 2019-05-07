# -*- coding: utf-8 -*-
"""

@author: Shreyas Menon
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import losses
from tensorflow.keras import backend
from tensorflow.keras import optimizers
import seaborn as sns

import cleaning 
import matplotlib.pyplot as plt


if __name__ == '__main__':
    

    df = pd.read_csv('knowledge_base.csv',index_col=0)
    
    df = df.reset_index(drop=True)
            
    df['query_request'] = cleaning.tokenize_lemmatize(list(df.query_request), extract_entities = False)
    
        
    df['intent'] = df.intent.apply(lambda x : str(x).lower())
    
    df_performance = pd.DataFrame(columns = ['optimizer','epochs','training_accuracy','testing_accuracy'])
    
    optimizers = ['sgd','adam','adagrad','rmsprop', 'adadelta']
    
    count = 0  
    for optimizer in optimizers:
        print("running for optimizer", optimizer)
        for epochs in range(1,21):
            count += 1
            encoder = LabelEncoder()
            topic_labels = encoder.fit_transform(df['intent'])
            
            encoder = OneHotEncoder(sparse=False)
            topic_labels = topic_labels.reshape((topic_labels.shape[0] , 1))  
            oht_encoder = encoder.fit_transform(topic_labels)
            
            NN_X_train, NN_X_test, NN_y_train, NN_y_test = train_test_split(df['query_request'], oht_encoder ,  test_size=0.2, random_state=0)
            
            #word embedding
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(NN_X_train)
            NN_X_train = tokenizer.texts_to_sequences(NN_X_train)
            NN_X_test = tokenizer.texts_to_sequences(NN_X_test)
            vocab_size = len(tokenizer.word_index) + 1
            
            #padding
            maxlen = 100
            NN_X_train  = pad_sequences(NN_X_train, padding='post', maxlen=maxlen)
            NN_X_test = pad_sequences(NN_X_test, padding='post', maxlen=maxlen)
            
            embedding_dim = 50
            input_dim =  vocab_size 
            
            nn_model = Sequential()
            
            nn_model.add(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen))
            nn_model.add(layers.GlobalMaxPool1D())
            nn_model.add(layers.Dense(512, activation='relu'))
            nn_model.add(layers.Dense(9, activation='sigmoid'))
            
            #nn_model.compile(loss='categorical_crossentropy', optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),  metrics=['accuracy'])   
           
            nn_model.compile(loss='categorical_crossentropy', optimizer= optimizer , metrics=['accuracy'])   
            nn_model.summary()
            
            #optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            history = nn_model.fit(NN_X_train, NN_y_train  , epochs = epochs , verbose=True, validation_data=(NN_X_test, NN_y_test))
            
            loss, train_accuracy = nn_model.evaluate(NN_X_train, NN_y_train, verbose=True)
            print("Training Accuracy: {:.4f}".format(train_accuracy))
            loss, test_accuracy = nn_model.evaluate(NN_X_test, NN_y_test, verbose=True)
            print("Testing Accuracy:  {:.4f}".format(test_accuracy))
            
            df_performance.loc[count] = [optimizer, epochs, train_accuracy, test_accuracy] 

    #uncomment for peformance only for epochs
    #df_performance = df_performance.set_index('optimizer',drop=True)
    #ax = dfp2.plot.line(figsize=(10,6) , title = "Neural network performance with epochs")
    #fig = ax.get_figure()
    #fig.savefig('neural_net_performance.jpg')
    
    df_pivot = df_performance.pivot(index='epochs', columns='optimizer', values='testing_accuracy')
    
    ax = df_pivot.plot(figsize=(10,6) ,title = "Testing accuracy for all optimizers")
    fig = ax.get_figure()
    fig.savefig('neural_net_performance_with_optimizers.jpg')
    df_performance.to_excel("NN_performance_with_optimizers.xlsx")
    
    
    
    