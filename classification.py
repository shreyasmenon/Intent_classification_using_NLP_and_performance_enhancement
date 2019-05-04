from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , StratifiedKFold


X_train, X_test, y_train, y_test = train_test_split(df['query'], df['topic'], test_size=0.2, random_state=0)

#Support vector machines
svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)
predicted= svc_model.predict(X_test)
print(classification_report(y_test,predicted))


