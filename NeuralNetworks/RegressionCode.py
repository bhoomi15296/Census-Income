#Guiding Question 1:
#1.1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()
fields=["age","weeks worked in year","income","wage per hour"]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)

df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)


final_val = df

cat_values = final_val[['income']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', random_state=np.random)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)


#1.2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()
fields=["age","weeks worked in year","income","wage per hour"]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)holder'])]

df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)


final_val = df

cat_values = final_val[['income']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,5,7), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='invscaling',momentum=0.9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#1.3
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()
fields=["age","weeks worked in year","income","wage per hour"]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)holder'])]

df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)


final_val = df

cat_values = final_val[['income']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,10), activation='identity', solver='lbfgs', alpha=0.0001, batch_size='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#Guiding Question 2
#2.1

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","income","education","marital status"]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)

final_val = df

labelEncode = final_val[['education']]
le = LabelEncoder()
labelEncod_enc = pd.DataFrame(scale(le.fit_transform(labelEncode)))
final_val.drop(['education'],axis=1,inplace=True)
final_val=pd.concat([final_val, labelEncod_enc], axis=1,sort=False)


cat_values = final_val[['income','marital status']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['income', 'marital status'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,5), activation='tanh', solver='sgd', alpha=0.0001, batch_size=100, learning_rate='adaptive',momentum=0.7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))
    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#2.2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","income","education","marital status"]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)

final_val = df

labelEncode = final_val[['education']]
le = LabelEncoder()
labelEncod_enc = pd.DataFrame(scale(le.fit_transform(labelEncode)))
final_val.drop(['education'],axis=1,inplace=True)
final_val=pd.concat([final_val, labelEncod_enc], axis=1,sort=False)


cat_values = final_val[['income','marital status']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['income', 'marital status'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,), activation='logistic', solver='lbfgs', alpha=0.0005, batch_size='auto', early_stopping=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))
    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#2.3
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","income","education","marital status"]

data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)

final_val = df

labelEncode = final_val[['education']]
le = LabelEncoder()
labelEncod_enc = pd.DataFrame(scale(le.fit_transform(labelEncode)))
final_val.drop(['education'],axis=1,inplace=True)
final_val=pd.concat([final_val, labelEncod_enc], axis=1,sort=False)


cat_values = final_val[['income','marital status']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['income', 'marital status'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(5,5,2), activation='relu', solver='adam', alpha=0.0007, batch_size=200, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))
    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#Guiding Question 3
#3.1
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","detailed household summary in household"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df=data.loc[data["detailed household summary in household"].isin([' Householder'])]
df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)


final_val = df

cat_values = final_val[['detailed household summary in household']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['detailed household summary in household'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,5,2,2), activation='identity', solver='adam', alpha=0.0001, batch_size='auto', beta_1=0.7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#3.2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","detailed household summary in household"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df=data.loc[data["detailed household summary in household"].isin([' Householder'])]
df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)


final_val = df

cat_values = final_val[['detailed household summary in household']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['detailed household summary in household'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(10,2), activation='logistic', solver='lbfgs', alpha=0.0005, batch_size=100, random_state=np.random)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

#3.3
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","detailed household summary in household"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
df=data.loc[data["detailed household summary in household"].isin([' Householder'])]
df = pd.DataFrame(data)


target = df['age']
df = df.drop(['age'], axis=1)

target=scale(target)
target = pd.DataFrame(target)


final_val = df

cat_values = final_val[['detailed household summary in household']]
cat_values_enc=pd.get_dummies(cat_values)
final_val.drop(['detailed household summary in household'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(scale(categorical_variable_encoded))


kf = KFold(n_splits=20)
folds = 20

loss = []
meanList = []
coeff = []
meanSquareList = []

i = 0
for train_index, test_index in kf.split(categorical_variable_encoded):
    i+=1
    print("Iteration: ",i)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = MLPRegressor(hidden_layer_sizes=(3,3,2,3,2), activation='tanh', solver='sdf', alpha=0.0002, batch_size='auto', early_stopping=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss_y = clf.loss_
    loss.append(loss_y)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

    
totalLoss=0
totalMean=0
totalRScore=0
totalMeanSquare=0

for i in range(len(loss)):
    totalLoss+= loss[i]
    totalMean+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Loss: ", totalLoss/folds)
print("Average Mean: ", totalMean/folds)
print("Average Coefficient: ",totalRScore/folds)
print("Average Root Mean Square Error: ",totalMeanSquare/folds)

print (datetime.now() - startTime)

