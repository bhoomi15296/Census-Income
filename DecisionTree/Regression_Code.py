#1 - Decision Tree on Guiding Question 1
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()


fields = ["age","capital gains","tax filer status"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

target = df['age']
df = df.drop(['age'], axis=1)

final_val = df
target=scale(target)
target = pd.DataFrame(target)

cat_values = final_val[['tax filer status']]

ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.get_dummies(cat_values)

final_val.drop(['tax filer status'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)

categorical_variable_encoded=scale(categorical_variable_encoded)
categorical_variable_encoded = pd.DataFrame(categorical_variable_encoded)

kf = KFold(n_splits=10)

meanList = []
coeff = []
meanSquareList = []

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    regressionTree = DecisionTreeRegressor(random_state=np.random, min_weight_fraction_leaf=0.05)
    regressionTree.fit(X_train, y_train)
    y_pred = regressionTree.predict(X_test)
    
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    print("Mean for each iteration: ", mean)
    print("Coefficient for each iteration: ",rScore)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

total=0
totalRScore=0
totalMeanSquare=0
for i in range(len(meanList)):
    total+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Mean: ", total/10)
print("Average Coefficient: ",totalRScore/10)
print("Average Root Mean Square Error: ",totalMeanSquare/10)

print(regressionTree.get_depth())

print (datetime.now() - startTime)


#2 - Linear Regression on Guiding Question 1

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from matplotlib.colors import ListedColormap
scaler = StandardScaler() 
 
fields = ["marital status","sex"]
predicter=["age"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
target=pd.DataFrame(preprocessing.scale(pd.read_csv('census-income.csv',skipinitialspace=True,usecols=predicter)))
df = pd.DataFrame(data, columns=fields)
res=pd.DataFrame(preprocessing.scale(pd.get_dummies(df)))

from sklearn.linear_model import LinearRegression
kf = KFold(n_splits=10)

meanList = []
coeff = []
meanSquareList = []
for train_index, test_index in kf.split(res):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = res.iloc[train_index], res.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
    reg=LinearRegression()
    reg.fit(x_train,y_train)
    
    y_pred= reg.predict(x_test)
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    print("Mean for each iteration: ", mean)
    print("Coefficient for each iteration: ",rScore)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))
total=0
totalRScore=0
totalMeanSquare=0
for i in range(len(meanList)):
    total+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Mean: ", total/10)
print("Average Coefficient: ",totalRScore/10)
print("Average Root Mean Square Error: ",totalMeanSquare/10)

#3 - ZeroR on Guiding Question 2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import statistics
from sklearn.metrics import r2_score

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from datetime import datetime
fields = ["age"]
startTime = datetime.now()
data=pd.read_csv('census-income.csv',skipinitialspace=True, usecols=fields)
df = pd.DataFrame(preprocessing.scale(data))

meanList = []
coeff = []
meanSquareList = []

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(df):
    target1=[]
    print("TRAIN:", train_index, "TEST:", test_index)
    y_train, y_test = df.iloc[train_index], df.iloc[test_index]
    for i in range(0,y_test.shape[0]):    
        target1.append(df.mean())
    mean = mean_absolute_error(y_test,target1)
    meanSquare = mean_squared_error(y_test, target1)
    rScore = r2_score(y_test, target1)
    print("Mean for each iteration: ", mean)
    print("Coefficient for each iteration: ",rScore)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))
total=0
totalRScore=0
totalMeanSquare=0
for i in range(len(meanList)):
    total+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Mean: ", total/10)
print("Average Coefficient: ",totalRScore/10)
print("Average Root Mean Square Error: ",totalMeanSquare/10)


print (datetime.now() - startTime)

#4 - Decision Tree on Guiding Question 2
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import scale

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()

fields = ["age","marital status","income","education"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

target = df['age']
df = df.drop(['age'], axis=1)

final_val = df

labelEncode = final_val[['education']]

le = LabelEncoder()

labelEncod_enc = pd.DataFrame(le.fit_transform(labelEncode))


final_val.drop(['education'],axis=1,inplace=True)

final_val=pd.concat([final_val, labelEncod_enc], axis=1,sort=False)

cat_values = final_val[['marital status','income']]

ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.get_dummies(cat_values)

final_val.drop(['marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)

kf = KFold(n_splits=10)

meanList = []
coeff = []
meanSquareList = []

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    regressionTree = DecisionTreeRegressor(random_state=0, max_features=2)
    regressionTree.fit(X_train, y_train)
    y_pred = regressionTree.predict(X_test)
    
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    print("Mean for each iteration: ", mean)
    print("Coefficient for each iteration: ",rScore)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

total=0
totalRScore=0
totalMeanSquare=0
for i in range(len(meanList)):
    total+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Mean: ", total/10)
print("Average Coefficient: ",totalRScore/10)
print("Average Root Mean Square Error: ",totalMeanSquare/10)

print(regressionTree.get_depth())

print ("Time taken to build the model: ",datetime.now() - startTime)



#5 - Model Tree on Guiding Question 2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from datetime import datetime

def split_data(j_feature, threshold, X, y):
        X = np.array(X)
        idx_left = np.where(X[:, j_feature] <= threshold)[0]
        idx_right = np.delete(np.arange(0, len(X)), idx_left)
        assert len(idx_left) + len(idx_right) == len(X)
        return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])

startTime = datetime.now()
fields = ["age","education","income","marital status"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

target = df['age']
df = df.drop(['age'], axis=1)

final_val = df
labelEncode = final_val[['education']]
le = LabelEncoder()
labelEncod_enc = pd.DataFrame(le.fit_transform(labelEncode))
final_val.drop(['education'],axis=1,inplace=True)
final_val=pd.concat([final_val, labelEncod_enc], axis=1,sort=False)

cat_values = final_val[['marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())
final_val.drop(['marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)

meanList = []
coeff = []

X_train, X_test, y_train, y_test = train_test_split(categorical_variable_encoded, target, test_size=0.3, random_state=0)     
regressionTree = DecisionTreeRegressor(random_state=0,min_samples_split=2,max_leaf_nodes=5)
regressionTree.fit(X_train, y_train)

n_nodes = regressionTree.tree_.node_count
children_left = regressionTree.tree_.children_left
print(children_left)
children_right = regressionTree.tree_.children_right
print(children_right)
feature = regressionTree.tree_.feature
threshold = regressionTree.tree_.threshold
leaves = np.arange(0,regressionTree.tree_.node_count)


thistree = [regressionTree.tree_.feature.tolist()]
thistree.append(regressionTree.tree_.threshold.tolist())
thistree.append(regressionTree.tree_.children_left.tolist())
thistree.append(regressionTree.tree_.children_right.tolist())

leaf_observations = np.zeros((n_nodes,len(leaves)),dtype=bool)

print(n_nodes," ",children_left," ",children_right," ",feature," ",threshold)

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
leafNodes = np.zeros(shape=n_nodes, dtype=np.int64)
stack = [(0, -1)] 
X_left=[]
y_left=[]
X_right=[]
y_right = []
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    print(node_id," ",parent_depth)
    node_depth[node_id] = parent_depth + 1
    
    print("left and right: ",node_id,": ",children_left[node_id]," ",children_right[node_id])
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))                 
    else:
        is_leaves[node_id] = True
        print(is_leaves)
        
def ModelTree():
    N = X_train.shape
    linearRegBest = np.Infinity
    meanSquareBest = 0
    for i in range(n_nodes):
        if is_leaves[i] == True:        
            continue
        else:         
            (X_left, y_left), (X_right, y_right) = split_data(feature[i], threshold[i], X_train, y_train)
            N_left, N_right = len(X_left), len(X_right)   
            
            X_left = np.array(X_left, dtype=np.float64)
            y_left = np.array(y_left, dtype=np.float64)
            X_left = pd.DataFrame(X_left)
            X_left = X_left.fillna(0)
            y_left = pd.DataFrame(y_left)
            y_left = y_left.fillna(0)
            
            X_right = np.array(X_right, dtype=np.float64)
            y_right = np.array(y_right, dtype=np.float64)
            X_right = pd.DataFrame(X_right)
            X_right = X_right.fillna(0)
            y_right = pd.DataFrame(y_right)
            y_right = y_right.fillna(0)
        
            reg = LinearRegression()
            
            leftModel =  reg.fit(X_left,y_left)
            rightModel = reg.fit(X_right, y_right)
            y_predLeft = reg.predict(X_left)
            y_predRight = reg.predict(X_right)
            lossLeft = mean_absolute_error(y_left, y_predLeft)
            lossRight = mean_absolute_error(y_right, y_predRight)
            meanSquareLeft = mean_squared_error(y_left, y_predLeft)
            meanSqaureRight = mean_squared_error(y_right, y_predRight)
            linearReg = (N_left*lossLeft + N_right*lossRight) / N 
            meanSquare = (N_left*meanSquareLeft + N_right*meanSqaureRight) / N
            linearReg = linearReg[0]
            meanSquare = np.sqrt(meanSquare[0])
            if(linearReg < linearRegBest):
                linearRegBest = linearReg
                meanSquareBest = meanSquare
                print(leftModel)
                model = [leftModel, rightModel]
                print(model)
    return linearReg, meanSquareBest

modelTree, meanSquare = ModelTree();
print(modelTree," ",meanSquare)
print ("Time taken to build the model: ",datetime.now() - startTime)

node_indicator = regressionTree.decision_path(X_test)

leave_id = regressionTree.apply(X_test)

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

#6 - Decision Tree on Guiding Question 3
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime

startTime = datetime.now()


fields = ["age","marital status","income","capital losses"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

df=df.loc[df["income"].isin(["-50000"])]
print(df)

target = df['age']
df = df.drop(['age'], axis=1)

final_val = df
print(final_val)

target=scale(target)
target = pd.DataFrame(target)


cat_values = final_val[['marital status','income']]
cat_values_enc=pd.get_dummies(cat_values)

final_val.drop(['marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)

categorical_variable_encoded=scale(categorical_variable_encoded)
categorical_variable_encoded = pd.DataFrame(categorical_variable_encoded)

print(categorical_variable_encoded)

kf = KFold(n_splits=10)

meanList = []
coeff = []
meanSquareList = []

for train_index, test_index in kf.split(categorical_variable_encoded):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    regressionTree = DecisionTreeRegressor(random_state=1, min_samples_leaf=10)
    regressionTree.fit(X_train, y_train)
    y_pred = regressionTree.predict(X_test)
    
    mean = mean_absolute_error(y_test,y_pred)
    meanSquare = mean_squared_error(y_test, y_pred)
    rScore = r2_score(y_test, y_pred)
    print("Mean for each iteration: ", mean)
    print("Coefficient for each iteration: ",rScore)
    meanList.append(mean)
    coeff.append(rScore)
    meanSquareList.append(np.sqrt(meanSquare))

total=0
totalRScore=0
totalMeanSquare=0
for i in range(len(meanList)):
    total+= meanList[i]
    totalRScore+=coeff[i]
    totalMeanSquare+=meanSquareList[i]
    
print("Average Mean: ", total/10)
print("Average Coefficient: ",totalRScore/10)
print("Average Root Mean Square Error: ",totalMeanSquare/10)

print(regressionTree.get_depth())

print (datetime.now() - startTime)
