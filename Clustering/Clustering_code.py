#kMeans: Guiding Question 1
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt  
import seaborn as sns

from collections import Counter

from datetime import datetime


def rename_columns(df, prefix='x'):
    df = df.copy()
    df.columns = [prefix + str(i) for i in df.columns]
    return df

startTime = datetime.now()

fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)


pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

kmeans = KMeans(n_clusters=4, init='kmeans++', n_init=5, max_iter=200, random_state=0, algorithm='auto')
kmeans.fit(X_pca)
y_method1 = kmeans.predict(X_pca)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print(method1_sse)
print(iteration)

print("Method1: ",Counter(kmeans.labels_))

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_method1, edgecolors='black', s=120)
plt.scatter(
    center[:, 0], center[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)

#kMeans: Guiding Question 1
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt  
import seaborn as sns

from collections import Counter

from datetime import datetime


def rename_columns(df, prefix='x'):
    df = df.copy()
    df.columns = [prefix + str(i) for i in df.columns]
    return df

startTime = datetime.now()

fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)


pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

kmeans = KMeans(n_clusters=4, init='random', n_init=5, max_iter=100, random_state=0, algorithm='full')
kmeans.fit(X_pca)
y_method1 = kmeans.predict(X_pca)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print(method1_sse)
print(iteration)

print("Method1: ",Counter(kmeans.labels_))

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_method1, edgecolors='black', s=120)
plt.scatter(
    center[:, 0], center[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)


#kMeans:Guiding Question 2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt  

from collections import Counter, defaultdict
from sklearn.metrics import silhouette_score

from datetime import datetime

startTime = datetime.now()
scaler = StandardScaler() 

fields = ["education","sex","class of worker","target."]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
data=data.loc[data["sex"].isin(['Female'])]

df = pd.get_dummies(data)
X=scaler.fit_transform(df)
print(df.size)
sse=[]
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state=1)
X = pca.fit_transform(X)



km = KMeans(n_clusters=3, algorithm='full', random_state=0)
y_km=km.fit_predict(X)
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue', edgecolor='black',
    label='cluster 3'
)

labels=km.labels_
# plot the centroids
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

method2_sse = km.inertia_
print(method2_sse)
s_score = silhouette_score(df, labels)

#kMeans:Guiding Question 2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt  

from collections import Counter, defaultdict
from sklearn.metrics import silhouette_score

from datetime import datetime

startTime = datetime.now()
scaler = StandardScaler() 

fields = ["education","sex","class of worker","target."]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)
data=data.loc[data["sex"].isin(['Female'])]

df = pd.get_dummies(data)
X=scaler.fit_transform(df)
print(df.size)
sse=[]
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state=1)
X = pca.fit_transform(X)



km = KMeans(n_clusters=3, algorithm='auto', random_state=0)
y_km=km.fit_predict(X)
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue', edgecolor='black',
    label='cluster 3'
)

labels=km.labels_
# plot the centroids
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

method2_sse = km.inertia_
print(method2_sse)
s_score = silhouette_score(df, labels)

#kMeans: Guiding Question 3
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = pd.get_dummies(target)
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

kmeans = KMeans(n_clusters=6, init='k-means++', n_init=5, max_iter=200, random_state=0, algorithm='auto')
kmeans.fit(X_pca)
y_method1 = kmeans.predict(X_pca)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

print("sse: ",method1_sse)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_method1, edgecolors='black', s=120)
plt.scatter(
    center[:, 0], center[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)

#kMeans: Guiding Question 3

import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = pd.get_dummies(target)
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

kmeans = KMeans(n_clusters=6, init='random', n_init=5, max_iter=100, random_state=0, algorithm='full')
kmeans.fit(X_pca)
y_method1 = kmeans.predict(X_pca)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

print("sse: ",method1_sse)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_method1, edgecolors='black', s=120)
plt.scatter(
    center[:, 0], center[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)

#Comparing Relative indices:
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from collections import Counter
from datetime import datetime


def rename_columns(df, prefix='x'):
    df = df.copy()
    df.columns = [prefix + str(i) for i in df.columns]
    return df

startTime = datetime.now()

fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)


pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)


kmeans = KMeans(n_clusters=4, init='k-means++', n_init=5, max_iter=200, random_state=0, algorithm='auto')
kmeans.fit(X_pca)
y_method1 = kmeans.predict(X_pca)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print(method1_sse)
print(iteration)

s_score = silhouette_score(X_pca, labels)
print(s_score)


kmeans = KMeans(n_clusters=4, init='random',max_iter=100, random_state=0, algorithm='full')
kmeans.fit(X_pca)
y_method2 = kmeans.predict(X_pca)
method2_sse = kmeans.inertia_
iteration = kmeans.n_iter_
print(method2_sse)
print(iteration)
labels = kmeans.labels_

y_adj = adjusted_rand_score(y_method1, y_method2)
y_norm = normalized_mutual_info_score(y_method1, y_method2)
y_mutual = adjusted_mutual_info_score(y_method1, y_method2)
print("adjusted_rand_score: ",y_adj)
print("normalized_mutual_info_score: ",y_norm)
print("adjusted_mutual_info_score: ",y_mutual)

#Hierarchical Guiding Question 1:
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime

startTime = datetime.now()


fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)

pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='single',compute_full_tree='auto',connectivity=None, distance_threshold=None)

y_method1 = cluster.fit_predict(X_pca)

print("Method1: ",Counter(y_method1))
print(cluster.n_connected_components_)

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

print (datetime.now() - startTime)

plt.scatter(X_pca[:,0],X_pca[:,1])
plt.show()

#Hierarchical Guiding Question 1:
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime

startTime = datetime.now()


fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)

pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = AgglomerativeClustering(n_clusters=5, affinity='cosine',linkage='complete',compute_full_tree='auto',connectivity=None, distance_threshold=None)

y_method1 = cluster.fit_predict(X_pca)

print("Method1: ",Counter(y_method1))
print(cluster.n_connected_components_)

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

print (datetime.now() - startTime)

plt.scatter(X_pca[:,0],X_pca[:,1])
plt.show()


#Hierarchical Guiding Question 2:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distance_threshold_,counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



from collections import Counter

from datetime import datetime

startTime = datetime.now()
import seaborn as sns; sns.set(color_codes=True)
fields = ["education", "class of worker","target.","sex"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["sex"].isin(['Female'])]
df = pd.DataFrame(data)

categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=2000, random_state=0)

pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean',linkage='ward',compute_full_tree='auto',connectivity=None, distance_threshold=None)
#cluster.fit(X_pca)
y_method1 = cluster.fit_predict(X_pca)

print("Method1: ",Counter(y_method1))
print(cluster.n_connected_components_)

#Hierarchical Guiding Question 2:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distance_threshold_,counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

from collections import Counter
from datetime import datetime

startTime = datetime.now()
import seaborn as sns; sns.set(color_codes=True)
fields = ["education", "class of worker","target.","sex"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["sex"].isin(['Female'])]
df = pd.DataFrame(data)

categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=2000, random_state=0)

pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = AgglomerativeClustering(n_clusters=6, affinity='cosine',linkage='complete',compute_full_tree='auto',connectivity=None, distance_threshold=None)
#cluster.fit(X_pca)
y_method1 = cluster.fit_predict(X_pca)

print("Method1: ",Counter(y_method1))
print(cluster.n_connected_components_)

#Hierarchical Guiding Question 3:
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  
from collections import Counter
from datetime import datetime

startTime = datetime.now()

fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = pd.get_dummies(target)
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = AgglomerativeClustering(n_clusters=6, affinity='manhattan',linkage='average',compute_full_tree='auto',connectivity=None, distance_threshold=None)
y_method1 = cluster.fit_predict(X_pca)

print("Method1: ",Counter(y_method1))
print(cluster.n_connected_components_)

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

print (datetime.now() - startTime)

plt.scatter(X_pca[:,0],X_pca[:,1])
plt.show()

#Hierarchical Guiding Question 3:
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  
from collections import Counter
from datetime import datetime

startTime = datetime.now()

fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = pd.get_dummies(target)
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean',linkage='ward',compute_full_tree='auto',connectivity=None, distance_threshold=None)
y_method1 = cluster.fit_predict(X_pca)

print("Method1: ",Counter(y_method1))
print(cluster.n_connected_components_)

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

print (datetime.now() - startTime)

plt.scatter(X_pca[:,0],X_pca[:,1])
plt.show()

#Comparing Relative Indices:
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

kmeans = KMeans(n_clusters=6, init='k-means++', n_init=5, max_iter=200, random_state=0, algorithm='auto')
kmeans.fit(X_pca)
y_method1 = kmeans.predict(X_pca)
center = kmeans.cluster_centers_
method1_sse = kmeans.inertia_
iteration = kmeans.n_iter_

labels = kmeans.labels_
print("sse: ",method1_sse)

s_score = silhouette_score(X_pca, labels)
print(s_score)

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean',linkage='single',compute_full_tree='auto',connectivity=None, distance_threshold=None)
y_method2 = cluster.fit_predict(X_pca)

y_adj = adjusted_rand_score(y_method1, y_method2)
y_norm = normalized_mutual_info_score(y_method1, y_method2)
y_mutual = adjusted_mutual_info_score(y_method1, y_method2)
print("adjusted_rand_score: ",y_adj)
print("normalized_mutual_info_score: ",y_norm)
print("adjusted_mutual_info_score: ",y_mutual)

#DBSCAN Guiding Question 1:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime

startTime = datetime.now()

fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)

pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = DBSCAN(algorithm='auto', eps=0.16, leaf_size=10, metric='euclidean',metric_params=None, min_samples=30, n_jobs=None, p=None)
y_method1 = cluster.fit_predict(X_pca)

labels = cluster.labels_

noOfClusters = len(set(labels))
print(noOfClusters)

print("Method1: ",Counter(y_method1))

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

print (datetime.now() - startTime)

#DBSCAN Guiding Question 1:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime

startTime = datetime.now()

fields = ["age","sex","race","marital status","income"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)
df = pd.DataFrame(data)

final_val = df

df['age'] = scale(df['age'])

cat_values = final_val[['sex','race','marital status','income']]
ohe = OneHotEncoder(drop='first')
cat_values_enc=pd.DataFrame(ohe.fit_transform(cat_values).toarray())

final_val.drop(['sex','race','marital status','income'],axis=1,inplace=True)
categorical_variable_encoded=pd.concat([final_val,cat_values_enc],axis=1,sort=False)
categorical_variable_encoded=pd.DataFrame(categorical_variable_encoded)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)

pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = DBSCAN(algorithm='ball_tree', eps=0.16, leaf_size=10, metric='l1',metric_params=None, min_samples=30, n_jobs=None, p=None)
y_method1 = cluster.fit_predict(X_pca)

labels = cluster.labels_

noOfClusters = len(set(labels))
print(noOfClusters)

print("Method1: ",Counter(y_method1))

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

print (datetime.now() - startTime)

#DBSCAN Guiding Question 2:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
startTime = datetime.now()


fields = ["education", "class of worker","target.","sex"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["sex"].isin(['Female'])]

final_val = data

categorical_variable_encoded=pd.get_dummies(data)
categorical_variable_encoded=categorical_variable_encoded.sample(n=1000, random_state=0)

pca = MDS(n_components = 3)
X_pca = pca.fit_transform(categorical_variable_encoded)


cluster = DBSCAN(algorithm='brute', eps=0.04, metric='cosine',metric_params=None, min_samples=100, n_jobs=None, p=None)
y_method1 = cluster.fit_predict(X_pca)

labels = cluster.labels_

noOfClusters = len(set(labels))
print(noOfClusters)

print("Method1: ",Counter(y_method1))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:,2], c=y_method1)

#DBSCAN Guiding Question 2:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
startTime = datetime.now()


fields = ["education", "class of worker","target.","sex"]
data=pd.read_csv('census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["sex"].isin(['Female'])]

final_val = data

categorical_variable_encoded=pd.get_dummies(data)
categorical_variable_encoded=categorical_variable_encoded.sample(n=1000, random_state=0)

pca = MDS(n_components = 3)
X_pca = pca.fit_transform(categorical_variable_encoded)


cluster = DBSCAN(algorithm='brute', eps=0.04, metric='cosine',metric_params=None, min_samples=100, n_jobs=None, p=None)
y_method1 = cluster.fit_predict(X_pca)

labels = cluster.labels_

noOfClusters = len(set(labels))
print(noOfClusters)

print("Method1: ",Counter(y_method1))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:,2], c=y_method1)

#DBSCAN Guiding Question 3:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime

startTime = datetime.now()


fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = DBSCAN(algorithm='ball_tree', eps=0.018, leaf_size=10, metric='manhattan',metric_params=None, min_samples=300, n_jobs=None, p=None)
y_method1 = cluster.fit_predict(X_pca)

labels = cluster.labels_

noOfClusters = len(set(labels))
print(noOfClusters)

print("Method1: ",Counter(y_method1))

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

homogeneity_score=homogeneity_score(target, labels)
v_measure_score=v_measure_score(target, labels, beta=20.0)
completeness_score = completeness_score(target, labels)
contingency_matrix = contingency_matrix(target, labels)

print("homogeneity_score: ",homogeneity_score)
print("v_measure_score: ",v_measure_score)
print("completeness_score: ",completeness_score)
print("contingency_matrix: ",contingency_matrix)

print (datetime.now() - startTime)

#DBSCAN Guiding Question 3:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, scale
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt  

from collections import Counter

from datetime import datetime

startTime = datetime.now()


fields = ["state of previous residence", "citizenship","country of birth self"]
data=pd.read_csv('D:\\WPI\\DataSets\\census-income.csv',skipinitialspace=True,usecols=fields)

data=data.loc[data["citizenship"].isin(['Native- Born in the United States', 'Foreign born- Not a citizen of U S '])]
df = pd.DataFrame(data)

df["state of previous residence"] = df["state of previous residence"].replace('?',df["state of previous residence"].mode()[0])
df["country of birth self"] = df["country of birth self"].replace('?',df["country of birth self"].mode()[0])
target=df['citizenship']
target = target.sample(n=20000, random_state=0)

df=df.drop(['citizenship'],axis=1)
categorical_variable_encoded=pd.get_dummies(df)
categorical_variable_encoded=categorical_variable_encoded.sample(n=20000, random_state=0)
pca = PCA(n_components = 2, random_state=1)

X_pca = pca.fit_transform(categorical_variable_encoded)

cluster = DBSCAN(algorithm='kd_tree', eps=0.018, leaf_size=10, metric='l1',metric_params=None, min_samples=300, n_jobs=None, p=None)
y_method1 = cluster.fit_predict(X_pca)

labels = cluster.labels_

noOfClusters = len(set(labels))
print(noOfClusters)

print("Method1: ",Counter(y_method1))

s_score = silhouette_score(X_pca, y_method1)
print(s_score)

homogeneity_score=homogeneity_score(target, labels)
v_measure_score=v_measure_score(target, labels, beta=20.0)
completeness_score = completeness_score(target, labels)
contingency_matrix = contingency_matrix(target, labels)

print("homogeneity_score: ",homogeneity_score)
print("v_measure_score: ",v_measure_score)
print("completeness_score: ",completeness_score)
print("contingency_matrix: ",contingency_matrix)

print (datetime.now() - startTime)
