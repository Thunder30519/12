#!/usr/bin/env python
# coding: utf-8

# In[2]:


### unsupevised learning ###
#太多的特征可能导致过拟合，用pca防止过拟合
#为了更好的可视化，因为把他转变成了2/3维度
#变量之间没有关系

#坏处
#稀少数据无法处理
#数据过于不相关，无法处理
#主要成分是垂直的
#新数据可能没有意义


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target 

print(X)

import numpy as np

mean_vec = np.mean(X, axis=0)
central_vec = X - mean_vec
cov_mat_1 = central_vec.T.dot(central_vec) / (X.shape[0] - 1)
cov_mat_2 = np.cov(central_vec)

eigvalues, eigvectors = np.linalg.eig (cov_mat_1)
print(eigvalues)
print(eigvectors)

eig_pair = [(np.abs(eigvalues[i]),eigvectors[:,i]) for i in range(len(eigvalues))]
eig_pair.sort(key=lambda x:x[0], reverse = True)

line = 0.99
alltogether = sum(np.abs(eigvalues))
now = 0
i = 0 
while now < line:  
    now += eigvalues[i]/alltogether
    print(now)
    i += 1
print(i-1)

num_features = X.shape[1]
proj_mat = eig_pair[0][1].reshape(num_features,1)
print(proj_mat, ' proj mat')
for eig_vec_idx in range(1, i):
    proj_mat = np.hstack((proj_mat, eig_pair[eig_vec_idx][1].reshape(num_features,1)))
print(proj_mat, ' proj mat')
pca_data = X.dot(proj_mat)
print(pca_data, ' pca data')


# In[3]:


from sklearn.decomposition import PCA
pca = PCA (n_components = 4)
pca.fit(X)
print(pca.explained_variance_ratio_, ' explained_variance_ratio_')
print(pca.singular_values_, ' singular_values_')


# In[31]:


import matplotlib.pyplot as plt


# In[52]:


plt.plot(["1","2","3","4"],eigvalues,"ro-")
plt.axhline(eigvalues[1])
plt.title('Scree plot')
plt.xlabel('N_component')
plt.ylabel('Eignvalues')
plt.show()


# In[53]:


### SVD ###
#A = UE(V^T)
#A^T = VE(U^T)
# AA^T = UE^2(U^T)
# A^TA = VE^2(V^T)


# In[103]:


A = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [11,12,13,14,15,16,17,18,19,20],
    [21,22,23,24,25,26,27,28,29,30]])
A = A - A.mean(axis=0)
U,s,VT = np.linalg.svd(A)
c1 = VT.T[:, 0]
c2 = VT.T[:, 1]
print(c1,"***********")
print(c2,"***********")
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[0],:A.shape[0]] = np.diag(s)
print(A)
W2 = VT.T[:, :2]
X2D = A.dot(W2)
print(X2D,"ssssssssss")
print(s**2/sum(s**2),'2222')



from sklearn.decomposition import TruncatedSVD
# define array
A = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [11,12,13,14,15,16,17,18,19,20],
    [21,22,23,24,25,26,27,28,29,30]])
print(A)
# svd
svd = TruncatedSVD(n_components = 3)
svd.fit(A)


# In[4]:


### clustering ###

### 步骤
# 设置k值，即集合的数量
# 选择k个集合的中心点（mean）
# 把某个k点最近的x，归类于第k个中心（全部）
# 新的x，可以合成一个新的k中心

## 分类的依据是让属于该集合的点到集合的中心的距离
## WC 是每个集合内的距离和
## WCSS 是所以集合的距离和


# In[46]:


# Common imports
import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
# Let's start by generating some blobs ---
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7) 
print(X, ' is X')
k = 5
kmeans = KMeans(n_clusters=k, n_init = 10)
y_labels = kmeans.fit_predict(X)
print('y_labels: ', y_labels)
plt.scatter(X[:,0],X[:,1], c=y_labels, cmap='rainbow')
plt.show()

minibatch_kmeans = MiniBatchKMeans(n_clusters=5,n_init = 10)
minibatch_kmeans.fit(X)
y_labels = kmeans.fit_predict(X)
print('y_labels: ', y_labels)
plt.scatter(X[:,0],X[:,1], c=y_labels, cmap='rainbow')
plt.show()


# In[54]:


### optimal clusters Silhouette diagrams###
import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# Let's start by generating some blobs ---
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
# ------- end of generating blobs ----- 
scores = []
for k in range(2,9):
    kmeans = KMeans(n_clusters=k,n_init = 10)
    y_pred = kmeans.fit_predict(X)
    score = silhouette_score(X, kmeans.labels_)
    scores.append(score)
plt.plot([i for i in range(2,9)], scores)
plt.ylabel = ("Silhouette Score")
plt.show()
print(' silhouette_scores: \n', scores)
score_max = np.max(scores)
k_max = np.argmax(scores)  + 2
print('k_max: ', k_max, '\nscore_max: ' , score_max)


# In[76]:


### silhouette diagram for each type ###
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1) 

range_n_clusters = [2, 3, 4, 5]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

clusterer = KMeans(n_clusters=2, n_init="auto", random_state=10)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)

y_lower = 10
for i in range(2):
    color = cm.nipy_spectral(float(i) / 2)
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_ylim([0,  len(X) + (3) * 10])
    y_lower = y_upper + 10 

clusterer = KMeans(n_clusters=3, n_init="auto", random_state=10)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)

y_lower = 10
for i in range(3):
    color = cm.nipy_spectral(float(i) / 3)
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    ax2.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax2.set_ylim([0, len(X) + (4) * 10])
    y_lower = y_upper + 10 

clusterer = KMeans(n_clusters=4, n_init="auto", random_state=10)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
y_lower = 10
for i in range(4):
    color = cm.nipy_spectral(float(i) / 4)
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    ax3.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    ax3.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax3.set_ylim([0, len(X) + (5) * 10])
    y_lower = y_upper + 10 

clusterer = KMeans(n_clusters=5, n_init="auto", random_state=10)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
y_lower = 10
for i in range(5):
    color = cm.nipy_spectral(float(i) / 5)
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    ax4.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    ax4.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax4.set_ylim([0,len(X) + (6) * 10])
    y_lower = y_upper + 10 
plt.show()


# In[ ]:





# In[ ]:





# In[3]:


### stacking ###

# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    return X, y

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

# get a list of models to evaluate
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    models['stacking'] = get_stacking()
    return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.savefig('stacked.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


### agglomerative clustering ###
### use similarity to cluster
### agglomerative clustering is good at small and diviseive clustering is good at large
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#The next step is to import or create the dataset. In this example, we'll use the following example data:

X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
cluster.fit_predict(X) 
print(cluster.labels_)


plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()


# In[36]:


### density-based clustering ###
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(n_samples=1000, noise=0.05)
print(X)
print(y)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
plt.scatter(X[:,0],X[:,1],c=dbscan.labels_)
print("accuracy_score: ", accuracy_score(y, dbscan.labels_))
print('dbscan.labels_[:10]: ', dbscan.labels_[:10])
print('np.unique(dbscan.labels_): ', np.unique(dbscan.labels_))


# In[32]:





# In[ ]:





# In[ ]:





# In[13]:


from sklearn.preprocessing import Normalizer
import sklearn.metrics import precision_recall_curve
T = Normalizer().fit(X)
T.transform(X)


# In[2]:


### auc p-f curve ###
train_x, train_y, test_x, test_y = train_test_split(x,y,test_size = 0.4)

n_prob = [0 for _ in range(len(test_y))]

model = ### name(parameter)
model.fit(train_x,train_y)

l_prob = model.predict_proba(test_x)
lr_prob = lr_prob[:, 1] ### only consider the positive part
ns_auc = roc_auc_score(testy, n_prob)
lr_auc = roc_auc_score(testy, l_prob)

print(ns_auc,lr_auc)

ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
#lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs) if precision and recall plot is drawed


plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, ns_tpr, linestyle = ".", label = 'Model')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.savefig('plot.png')


# In[1]:


### 独热编码或者label编码 ###
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)


# In[71]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
print(breast_cancer_wisconsin_original)
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_original.variables) 


# In[72]:


breast_cancer_wisconsin_original['data']['features']


# In[73]:


breast_cancer_wisconsin_original['data']['targets']


# In[81]:


feature = breast_cancer_wisconsin_original['data']['features'].values
y = breast_cancer_wisconsin_original['data']['targets'].values


# In[88]:


feature = breast_cancer_wisconsin_original['data']['features'].values
y = breast_cancer_wisconsin_original['data']['targets'].values
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
X = feature
T = MinMaxScaler().fit(X)
X = T.transform(X)
new_array = np.delete( np.array([i for i in range(len(X))]), np.where(np.isnan(X))[0])
X = X[new_array]
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y)
y = y[new_array]

from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy = train_test_split(X, y, test_size = 0.4)
model = MLPClassifier(hidden_layer_sizes = (30,20,10), solver = "adam", learning_rate_init = 0.1)
model.fit(trainx, trainy)
model.predict(testx)
acc = accuracy_score(testy, model.predict(testx))
print(acc)
rmse = np.sqrt(mean_squared_error(testy, model.predict(testx)))
print(rmse)

model = MLPClassifier(hidden_layer_sizes = (30,20,10), solver = "sgd", learning_rate_init = 0.1)
model.fit(trainx, trainy)
model.predict(testx)
acc = accuracy_score(testy, model.predict(testx))
print(acc)
rmse = np.sqrt(mean_squared_error(testy, model.predict(testx)))
print(rmse)


# In[119]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot  as plt
train_x, test_x, train_y, test_y = train_test_split(X,y,test_size = 0.4)

n_prob = [0 for _ in range(len(test_y))]

model = MLPClassifier(hidden_layer_sizes = (30,20,10), solver = "adam", learning_rate_init = 0.1)
model.fit(train_x,train_y)

l_prob = model.predict_proba(test_x)

lr_prob = l_prob[:, 1] 
ns_auc = roc_auc_score(testy[:,0], n_prob)
lr_auc = roc_auc_score(testy[:,0], l_prob[:,0])

print(ns_auc,lr_auc)

ns_fpr, ns_tpr, _ = roc_curve(testy[:,0], n_prob)
lr_fpr, lr_tpr, _ = roc_curve(testy[:,0], l_prob[:,0])
#lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs) if precision and recall plot is drawed


plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, label = 'Model')
plt.show()

#pyplot.xlabel('False Positive Rate')
#pyplot.ylabel('True Positive Rate')
#pyplot.legend()
#pyplot.savefig('plot.png')


# In[108]:


test_y[:,0]


# In[ ]:





# In[ ]:


##q1
import pandas as pd
import numpy as np

# Load dataset
# Create URL
url = './dataset.csv'
dataframe = pd.read_csv(url)
dataset = dataframe.values

dataframe.dropna(inplace = True) 

X = dataset[: , 0:-1]
y = dataset[: , -1]
'''
Task-1
'''

# code for Task-1 here
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
T = MinMaxScaler().fit(X)
X = T.transform(X)
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.4, random_state = 11)
model = RandomForestClassifier(n_estimators = 200 , criterion = 'gini', max_depth = 13, random_state = 2023)
model.fit(trainx, trainy)
pre = precision_score(testy, model.predict(testx))
rec = recall_score(testy, model.predict(testx))
print("precision: ", pre)  
print("Recall: ", rec)
#best_model_task1 = ??
best_model_task1 = RandomForestClassifier(n_estimators = 200 , criterion = 'gini', max_depth = 13, random_state = 2023)
#============== ==========  
'''
Task-2
'''
 

# code for Task-2 here
from sklearn.decomposition import PCA
k = 0 
i = 1
while k < 0.95:
    pca = PCA (n_components = i)
    pca.fit(X)
    k = sum(pca.explained_variance_ratio_)
    i += 1
print("Number of components is: ", i)
print("Variance ratio is: ", sum(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_, ' explained_variance_ratio_')
X = pca.fit_transform(X)

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.4, random_state = 11)
model = RandomForestClassifier(n_estimators = 300 , criterion = 'gini', max_depth = 12, random_state = 2023)
model.fit(trainx, trainy)
pre = precision_score(testy, model.predict(testx))
rec = recall_score(testy, model.predict(testx))
print("precision: ", pre)  
print("Recall: ", rec)


# In[ ]:





# In[ ]:


##q2
import pandas as pd
import numpy as np

# Load dataset
# Create URL
url = './dataset.csv'
dataframe = pd.read_csv(url)
dataset = dataframe.values

dataframe.dropna(inplace = True) 

X = dataset[: , 0:-1]
y = dataset[: , -1]
'''
Task-1
'''

# code for Task-1 here
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
T = MinMaxScaler().fit(X)
X = T.transform(X)
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.4, random_state = 11)
model = RandomForestClassifier(n_estimators = 200 , criterion = 'gini', max_depth = 13, random_state = 2023)
model.fit(trainx, trainy)
pre = precision_score(testy, model.predict(testx))
rec = recall_score(testy, model.predict(testx))
print("precision: ", pre)  
print("Recall: ", rec)
#best_model_task1 = ??
best_model_task1 = RandomForestClassifier(n_estimators = 200 , criterion = 'gini', max_depth = 13, random_state = 2023)
#============== ==========  
'''
Task-2
'''
 

# code for Task-2 here
from sklearn.decomposition import PCA
k = 0 
i = 1
while k < 0.95:
    pca = PCA (n_components = i)
    pca.fit(X)
    k = sum(pca.explained_variance_ratio_)
    i += 1
print("Number of components is: ", i)
print("Variance ratio is: ", sum(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_, ' explained_variance_ratio_')
X = pca.fit_transform(X)

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.4, random_state = 11)
model = RandomForestClassifier(n_estimators = 300 , criterion = 'gini', max_depth = 12, random_state = 2023)
model.fit(trainx, trainy)
pre = precision_score(testy, model.predict(testx))
rec = recall_score(testy, model.predict(testx))
print("precision: ", pre)  
print("Recall: ", rec)
#best_model_task2 = ??


#============== ==========  
'''
Task-3
First compare the performance of the above two models.
'''



''' 
Now, explain the difference below.



'''


# In[ ]:


### Bayesion distributino ###

