from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , StratifiedKFold

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras import optimizers


X_train, X_test, y_train, y_test = train_test_split(df['query_request'], df['intent'], test_size=0.2, random_state=0)

#Guassian NB
nb_model = GaussianNB()
nb_model.fit(X_train,y_train)
predicted = model.predict(X_test)
print(classification_report(y_test,predicted))

#Support vector machines
svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)
predicted= svc_model.predict(X_test)
print(classification_report(y_test,predicted))

#Neural Networks
encoder = LabelEncoder()
topic_labels = encoder.fit_transform(df['intent'])

encoder = OneHotEncoder(sparse=False)
topic_labels = topic_labels.reshape((topic_labels.shape[0] , 1))  
oht_encoder = encoder.fit_transform(topic_labels)

NN_X_train, NN_X_test, NN_y_train, NN_y_test = train_test_split(df['query'], oht_encoder ,  test_size=0.2, random_state=0)

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
nn_model.compile(loss='categorical_crossentropy', optimizer = optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0),  metrics=['accuracy'])   
nn_model.summary()


