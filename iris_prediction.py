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

print("now removing some features......\n")
feature=feature.drop("sepal_length",axis=1)
feature=feature.drop("sepal_width",axis=1)

label_train, label_test, feature_train, feature_test = train_test_split(label,feature, test_size=0.4, random_state=0)

feature_train=feature_train.astype(np.float)
feature_test=feature_test.astype(np.float)

clf = svm.SVC(kernel='linear', C=1).fit(feature_train, label_train)
svm_scr=clf.score(feature_test, label_test)
print("the SVM score is : ",svm_scr)

print("prediction: ", clf.predict([[1,1]]))