import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import precision_score,classification_report, confusion_matrix
dataset = pd.read_csv('testdataset.csv')
dataset = dataset.drop(columns='Unnamed: 0')
dataset = dataset.drop(columns='ReadingScore')
dataset = dataset.drop(columns='WritingScore')
dataset = dataset.drop(columns='EthnicGroup')
replace_gender ={'female':0,'male':1,np.NaN:-1}
dataset['Gender']=dataset['Gender'].map(replace_gender)
replace_PE={'some college':0,"bachelor's degree":1,np.NaN:-1,"master's degree":2,"associate's degree":3,"some high school":4,"high school":5}
dataset['ParentEduc']=dataset['ParentEduc'].map(replace_PE)
replace_lt ={'free/reduced':0,np.NaN:-1,'standard':1}
dataset['LunchType']=dataset['LunchType'].map(replace_lt)
replace_pmt ={'married':0,'single':1,np.NaN:-1,'divorced':2,'widowed':3}
dataset['ParentMaritalStatus']=dataset['ParentMaritalStatus'].map(replace_pmt)
replace_tp ={'none':0,'completed':1,np.NaN:-1}
dataset['TestPrep']=dataset['TestPrep'].map(replace_tp)
replace_ps ={'regularly':0,'sometimes':1,'never':2,np.NaN:-1}
dataset['PracticeSport']=dataset['PracticeSport'].map(replace_ps)
replace_ifc ={'yes':0,'no':1,np.NaN:-1}
dataset['IsFirstChild']=dataset['IsFirstChild'].map(replace_ifc)
replace_tm ={'school_bus':0,'private':1,np.NaN:-1}
dataset['TransportMeans']=dataset['TransportMeans'].map(replace_tm)
replace_wsh={'< 5':0,'5 - 10':1,'> 10':2,np.NaN:-1}
dataset['WklyStudyHours']=dataset['WklyStudyHours'].map(replace_wsh)
replace_ns ={np.NaN:-1,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}
dataset['NrSiblings']=dataset['NrSiblings'].map(replace_ns)
replace_math={0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:1,33:1,34:2,35:2,36:2,37:2,38:2,39:2,40:2,41:2,42:2,43:2,44:2,45:2,46:2,47:2,48:2,49:2,50:2,51:2,52:2,53:2,54:2,55:2,56:2,57:2,58:2,59:2,60:2,61:2,62:2,63:2,64:2,65:2,66:2,67:3,68:3,69:3,70:3,71:3,72:3,73:3,74:3,75:3,76:3,77:3,78:3,79:3,80:3,81:3,82:3,83:3,84:3,85:3,86:3,87:3,88:3,89:3,90:3,91:3,92:3,93:3,94:3,95:3,96:3,97:3,98:3,99:3,100:3,np.NaN:-1}
dataset['MathScore']=dataset['MathScore'].map(replace_math)
print(1)
X_train, X_test, Y_train, Y_test,=train_test_split(
dataset.iloc[:,:10],
dataset.iloc[:,10],
test_size = 0.5
)
print(1)
classifier = DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
plt.figure(figsize=(100,100))
print(1)
tree.plot_tree(classifier,fontsize=10)
plt.savefig('derevo1.pdf')
print(1)
print(1)
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(precision_score(Y_test,Y_pred,average='micro'))
print(1)
classifier = DecisionTreeClassifier(max_leaf_nodes=21)
classifier.fit(X_train,Y_train)
plt.figure(figsize=(40,20))
tree.plot_tree(classifier,fontsize=10)
print(1)
plt.savefig('derevo2.pdf')
print(1)
Y_pred = classifier.predict(X_test)
print(1)
print(confusion_matrix(Y_test,Y_pred))
print(precision_score(Y_test,Y_pred,average='micro'))
zm=0.25  
x=5
check=0
while check<1:
    classifier = DecisionTreeClassifier(max_leaf_nodes=x)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
    z= precision_score(Y_test,Y_pred,average='micro')
    if z> (zm-0.0000000000001):
zm=z
    else:
        check=1
    x = x+1
print(z,' ',x)
plt.figure(figsize=(40,20))
tree.plot_tree(classifier,fontsize=10)
plt.savefig('derevo3.pdf')
tree.plot_tree(classifier,fontsize=10)
plt.savefig('derevo3.pdf')
