#1 - ZeroR for Guiding Question 1
from sklearn.model_selection import KFold
import pandas as pd

from datetime import datetime

def zeroR(train):
    target = train[40]
    unique_values = []
    for variable in target:
        if variable not in unique_values:
            unique_values.append(variable)
    
    countList = []
    for i in unique_values:
        count=0
        for variable in target:
            if variable == i:
                count+=1
        countList.append(count)
    
    idx = countList.index(max(countList))
    return unique_values[idx],countList,idx
    

startTime = datetime.now()
fields = ["age","income","capital gains","divdends from stocks","weeks worked in year"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

target = pd.DataFrame(df[40])

accuracyList = []
totalCount=[]

kf = KFold(n_splits=10) 

for train_index, test_index in kf.split(df):
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    zeroROutput,countList,idx = zeroR(df)
    
    y_test = pd.DataFrame(y_test)
    
    count = 0;
    for i in range(len(y_test)):
        x = y_test.iloc[i,0]
        if x == zeroROutput:
            count+=1
    
    accuracy = (count/len(y_test))*100
    totalCount.append(count)
    accuracyList.append(accuracy)    

print("Accuracy:", sum(accuracyList)/10);
count = sum(totalCount)/10

truePositive = count;
falsePositive = len(y_test) - count

confusionMatrix = [truePositive, falsePositive]

print("Confusion Matrix: ",confusionMatrix)

recall = truePositive/truePositive

print("Recall: ", recall)

precision = truePositive/(truePositive + falsePositive)
print("Precision: ",precision)

print ("Time to Build Model: ",datetime.now() - startTime)

#2 - OneR for Guiding Question 1
import pandas as pd
from sklearn.model_selection import KFold
from datetime import datetime

class OneR(object):    
    def fit(self, X, y):
        max_accuracy=0
        output = list()
        result = dict()
        
        dfx = pd.DataFrame(X)
        
        for i in dfx:
            result[str(i)] = dict()
            data = pd.DataFrame({"attribute":dfx[i], "target":y})
            cross_table = pd.crosstab(data.attribute, data.target)
            summary = cross_table.idxmax(axis=1)
            result[str(i)] = dict(summary)
            
            counts = 0
            
            for idx, row in data.iterrows():               
                if row['target'] == result[str(i)][row['attribute']]:
                    counts += 1                

            accuracy = (counts/len(y))
            
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                
            
            
            result_feature = {"variable": str(i), "accuracy":accuracy, "rules": result[str(i)] }  
            output.append(result_feature)
            
        return output

startTime = datetime.now()
fields = ["age","income","capital gains","divdends from stocks","weeks worked in year"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

target = df['income']
df = df.drop(['income'],axis=1)

kf = KFold(n_splits=10) 

accuracyList=[]

for train_index, test_index in kf.split(df):
    print(train_index, " ", test_index)
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    y_train, y_test = target.iloc[train_index],df.iloc[test_index]
    clf = OneR()
    results = clf.fit(X_train, y_train)
    accuracyList.append(clf.max_accuracy)

sumAccuracy=0
for i in range(10):
    sumAccuracy+= accuracyList[i]    

accuracy = sumAccuracy/10

print(accuracy)
print (datetime.now() - startTime)


#3 - Decision Tree on Guiding Question 1
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


from sklearn.model_selection import KFold

from datetime import datetime

startTime = datetime.now()

fields = ["age","capital gains","weeks worked in year","divdends from stocks","income"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

target = df['income']
df = df.drop(['income'], axis=1)

final_val = df

le = LabelEncoder()
target = pd.DataFrame(le.fit_transform(target))


categorical_variable_encoded = df

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    print(train_index, " ", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=60)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print("AUC: ",sumAUC/10)

print(clf.get_depth())

print ("Time to Build Model: ",datetime.now() - startTime)

#4 - Decision Tree on Guiding Question 2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


from sklearn.model_selection import KFold

from datetime import datetime

startTime = datetime.now()

fields = ["age", "sex","marital status","education","target."]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["education"].isin(['Masters degree(MA MS MEng MEd MSW MBA)', 'Associates degree-occup /vocational', 'Associates degree-academic program', 'Doctorate degree(PhD EdD)', 'Prof school degree (MD DDS DVM LLB JD)'])]

target = pd.DataFrame(data['target.'])
data = data.drop(['target.'], axis=1)
df = pd.get_dummies(data)

le = LabelEncoder()

target = pd.DataFrame(le.fit_transform(target))

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]
from sklearn.ensemble import RandomForestClassifier

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    print(train_index, " ", test_index)
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = DecisionTreeClassifier(criterion = "entropy", random_state=0, max_depth=20)
    clf=clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print("AUC: ",sumAUC/10)


print (datetime.now() - startTime)

#5 - Random Forest on Guiding Question 2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


from sklearn.model_selection import KFold

from datetime import datetime

startTime = datetime.now()

fields = ["age", "sex","marital status","education","target."]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["education"].isin(['Masters degree(MA MS MEng MEd MSW MBA)', 'Associates degree-occup /vocational', 'Associates degree-academic program', 'Doctorate degree(PhD EdD)', 'Prof school degree (MD DDS DVM LLB JD)'])]

target = pd.DataFrame(data['target.'])
data = data.drop(['target.'], axis=1)
df = pd.get_dummies(data)

le = LabelEncoder()

target = pd.DataFrame(le.fit_transform(target))

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]
from sklearn.ensemble import RandomForestClassifier

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    print(train_index, " ", test_index)
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    clf = RandomForestClassifier(n_estimators=10, criterion = "entropy", random_state=0, min_samples_leaf=10)
    clf=clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print("AUC: ",sumAUC/10)


print (datetime.now() - startTime)

#6 - Random Forest on Guiding Question 3
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


from sklearn.model_selection import KFold

from datetime import datetime

startTime = datetime.now()

fields = ["class of worker", "industry code","occupation code","income"]

data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)


target = df['income']
df = df.drop(['income'], axis=1)


final_val = df

le = LabelEncoder()
target = pd.DataFrame(le.fit_transform(target))

cat_values = final_val[['class of worker']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['class of worker'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)

accuracyList=[]
precisionList=[]
recallList=[]
f1List=[]
AUCList=[]

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    print(train_index, " ", test_index)
    X_train, X_test = categorical_variable_encoded.iloc[train_index], categorical_variable_encoded.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]    
    
    clf = RandomForestClassifier(n_estimators=10, criterion = "entropy", random_state=0, max_features='auto')
    clf = clf.fit(X_train, y_train) 
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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
for i in range(10):
    sumAccuracy+=accuracyList[i]
    sumPrecision+=precisionList[i]
    sumRecall+=recallList[i]
    sumF1+=f1List[i]
    sumAUC+=AUCList[i]
    
print("Accuracy: ",sumAccuracy/10)
print("Precision: ",sumPrecision/10)
print("Recall: ",sumRecall/10)
print("f1: ",sumF1/10)
print("AUC: ",sumAUC/10)

for estimator in clf.estimators_:
    print(estimator.tree_.max_depth)


print ("Time to Build the Model: ",datetime.now() - startTime)
