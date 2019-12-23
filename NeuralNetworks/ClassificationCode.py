#classification on guiding question 1
#hidden_layer_sizes=(10,), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant', random_state=1

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime

startTime = datetime.now()

fields = ["age","capital gains","weeks worked in year","divdends from stocks","target."]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant', random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)


#classification on guiding question 1
#hidden_layer_sizes=(10,5), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant', random_state=1

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime

startTime = datetime.now()

fields = ["age","capital gains","weeks worked in year","divdends from stocks","target."]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant', random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)


#classification on guiding question 1
#hidden_layer_sizes=(10,5), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', random_state=1

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime

startTime = datetime.now()

fields = ["age","capital gains","weeks worked in year","divdends from stocks","target."]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)

#classification on guiding question 2
#hidden_layer_sizes=(10,5,2), activation='tanh', solver='sgd', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["own business or self employed","full or part time employment stat","occupation code","target."]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5,2), activation='tanh', solver='sgd', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)


#classification on guiding question 2
#hidden_layer_sizes=(10,5,2), activation='tanh', solver='sgd', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["own business or self employed","class of worker","full or part time employment stat","occupation code","wage per hour","target."]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5,2), activation='tanh', solver='sgd', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)



#classification on guiding question 2
#hidden_layer_sizes=(10,5,2), activation='tanh', solver='adam', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["own business or self employed","class of worker","full or part time employment stat","occupation code","wage per hour","target."]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5,2), activation='tanh', solver='adam', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)



#classification on guiding question 2
#hidden_layer_sizes=(10,5,2), activation='tanh', solver='lbfgs', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5

import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["own business or self employed","class of worker","full or part time employment stat","occupation code","wage per hour","target."]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5,2), activation='tanh', solver='lbfgs', alpha=0.0002, batch_size=200, learning_rate='constant', random_state=1, momentum=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)

# classification on guiding question 2
#hidden_layer_sizes=(10,5), activation='logistic', solver='lbfgs', alpha=0.0002, batch_size=400, learning_rate='constant', random_state=1, momentum=0.8
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["own business or self employed","class of worker","full or part time employment stat","occupation code","wage per hour","target."]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5), activation='logistic', solver='lbfgs', alpha=0.0002, batch_size=400, learning_rate='constant', random_state=1, momentum=0.8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)



#classification on guiding question 3
#(hidden_layer_sizes=(10,5), activation='tanh', solver='adam', alpha=0.0002, batch_size=250, learning_rate='constant', random_state=1
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["country of birth self","citizenship","hispanic Origin","race","target."]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5), activation='tanh', solver='adam', alpha=0.0002, batch_size=250, learning_rate='constant', random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)



#classification on guiding question 3
#(hidden_layer_sizes=(10,5), activation='tanh', solver='adam', alpha=0.0001, batch_size=500, learning_rate='constant', random_state=1
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
import warnings
warnings.filterwarnings('always')
startTime = datetime.now()

fields = ["country of birth self","citizenship","hispanic Origin","race","target."]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['target.']
df = df.drop(['target.'], axis=1)
le=LabelEncoder()
target= le.fit_transform(target)
scaler = StandardScaler() 
target = pd.DataFrame(target)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=pd.DataFrame(scaler.fit_transform(categorical_variable_encoded))

kf = KFold(n_splits=20)

loss = []

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

scaler = StandardScaler()

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPClassifier(hidden_layer_sizes=(10,5), activation='tanh', solver='adam', alpha=0.0001, batch_size=500, learning_rate='constant', random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y=clf.loss_
    loss.append(loss_y)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred,pos_label=1)
    recall = metrics.recall_score(y_test, y_pred,pos_label=1)
    f1 = metrics.f1_score(y_test,y_pred,pos_label=1) 
    auc = metrics.roc_auc_score(y_test,y_pred)
    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    AUCList.append(auc)

sumAccuracy=0
sumPrecision=0
sumRecall=0
sumF1=0
sumAUC=0    
totalLoss=0
for i in range(len(loss)):
    totalLoss+= loss[i]
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/20)
print("Precision: ",sumPrecision/20)
print("Recall: ",sumRecall/20)
print("f1: ",sumF1/20)
print("AUC: ",sumAUC/20) 
print("Average Loss: ", totalLoss/20)


print ("Time to Build Model: ",datetime.now() - startTime)