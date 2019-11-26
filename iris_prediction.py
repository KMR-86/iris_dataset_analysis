import iris_analysis as ir
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
label=ir.iris["species"]
feature=ir.iris.drop("species", axis=1)
#splitting the data
label_train, label_test, feature_train, feature_test = train_test_split(label,feature, test_size=0.4, random_state=0)
#converting string types into float type
feature_train=feature_train.astype(np.float)
feature_test=feature_test.astype(np.float)


print("feature_train",feature_train.shape)
print("label_train",label_train.shape)
print("feature_test",feature_test.shape)
print("label_test",label_test.shape)

from sklearn import svm

clf = svm.SVC(kernel='linear', C=1).fit(feature_train, label_train)
svm_scr=clf.score(feature_test, label_test)
print("the SVM score is : ",svm_scr)

print("prediction: ", clf.predict([[1,1,1,1]]))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(feature_train,label_train)
gnb_scr=gnb.score(feature_test, label_test)
print("the Guassian NB score is : ",gnb_scr)
print("prediction: ", gnb.predict([[1,1,1,1]]))


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(feature_train,label_train)
tree_scr=tree.score(feature_test, label_test)
print("the Decision Tree score is : ",tree_scr)
print("prediction: ", tree.predict([[1,1,1,1]]))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(feature_train,label_train)
rf_scr=rf.score(feature_test, label_test)
print("the random Forest score is : ",rf_scr)
print("prediction: ", rf.predict([[1,1,1,1]]))

from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier(n_estimators=100, random_state=0)
ab.fit(feature_train,label_train)
ab_scr=ab.score(feature_test, label_test)
print("the adaboost score is : ",ab_scr)
print("prediction: ", ab.predict([[1,1,1,1]]))


print("\n now removing some features......\n")


feature=feature.drop("sepal_length",axis=1)
feature=feature.drop("sepal_width",axis=1)

label_train, label_test, feature_train, feature_test = train_test_split(label,feature, test_size=0.4, random_state=0)


print("feature_train",feature_train.shape)
print("label_train",label_train.shape)
print("feature_test",feature_test.shape)
print("label_test",label_test.shape)


feature_train=feature_train.astype(np.float)
feature_test=feature_test.astype(np.float)

clf_svm = svm.SVC(kernel='linear', C=1).fit(feature_train, label_train)
svm_scr2=clf_svm.score(feature_test, label_test)
print("the modified SVM score is : ",svm_scr2)

print("prediction: ", clf_svm.predict([[1,1]]))

gnb2 = GaussianNB()
gnb2 = gnb2.fit(feature_train, label_train)
gnb_scr2=gnb2.score(feature_test, label_test)
print("the modified Gaussian NB score is : ",gnb_scr2)

print("prediction: ", gnb2.predict([[1,1]]))


tree2 = DecisionTreeClassifier()
tree2.fit(feature_train,label_train)
tree_scr2=tree2.score(feature_test, label_test)
print("the modified Decision Tree score is : ",tree_scr2)
print("prediction: ", tree2.predict([[1,1]]))

rf2 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf2.fit(feature_train,label_train)
rf_scr2=rf2.score(feature_test, label_test)
print("the random Forest score is : ",rf_scr2)
print("prediction: ", rf2.predict([[1,1]]))


ab2 = AdaBoostClassifier(n_estimators=100, random_state=0)
ab2.fit(feature_train,label_train)
ab_scr2=ab2.score(feature_test, label_test)
print("the modified adaboost score is : ",ab_scr2)
print("prediction: ", ab2.predict([[1,1]]))