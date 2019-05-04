from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , StratifiedKFold


X_train, X_test, y_train, y_test = train_test_split(df['query_request'], df['intent'], test_size=0.2, random_state=0)

#Guassian NB
nb_model = GaussianNB()
model.fit(X_train,y_train)
predicted = model.predict(X_test)
print(classification_report(y_test,predicted))

#Support vector machines
svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)
predicted= svc_model.predict(X_test)
print(classification_report(y_test,predicted))


