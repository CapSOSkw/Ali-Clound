import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("new_format_full_target.csv")
data = data.drop(['xid', 'yid', 'hour', 'date_id'], axis=1)

#_________________________________________________________________________________________#
#Prepare

X = data.iloc[:, :-1]
Y = data.iloc[:, -1:]>15

safe_points_num = (Y==False).sum()[0]
danger_points_num = (Y==True).sum()[0]

print("There are %s safe points in data." % safe_points_num)
print("There are %s dangerous points in data." % danger_points_num)

scaler = StandardScaler()
X = scaler.fit_transform(X)
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#__________________________________________________________________________________________#
# Linear Regression

# model_LinearReg = LinearRegression()
# model_LinearReg.fit(X_train,y_train)
#
# pred = model_LinearReg.predict(X_test)
#
# print("Mean squared error: %.2f"
#       % mean_squared_error(y_test, pred))
#
# print('Variance score: %.2f' % r2_score(y_test, pred))

'''
Linear Regression: MSE: 4.40, R2: 0.85

'''

#___________________________________________________________________________________________#
# Sklearn neural networks

# model_NN = MLPClassifier(hidden_layer_sizes=(64,128,256), activation='relu',
#                          solver='sgd', alpha=0.001, learning_rate_init=0.01,
#                          verbose=True, early_stopping=True, )
#
# model_NN.fit(X_train,y_train)
# print(model_NN.score(X_test, y_test))


'''
Too fucking slow.
'''
#____________________________________________________________________________________________#
# Tensor Flow Neural Networks

# feature_columns = [tf.feature_column.numeric_column("x", shape=[10])]
#
# best_accuracy = 0
# best_layers = -1
#
# # Add layers here:
# for i in np.array([[64,128,256]]):
#     classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
#                                             hidden_units=i,
#                                             n_classes=2,
#                                             optimizer=tf.train.ProximalAdagradOptimizer(
#                                                 learning_rate=0.01,
#                                                 l2_regularization_strength=0.001
#                                             ),
#                                             activation_fn=tf.nn.relu,
#                                         )
#
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#             x={"x": np.array(X_train)},
#             y=np.array(y_train),
#             num_epochs=None,
#             shuffle=True)
#
#     classifier.train(input_fn=train_input_fn, steps=2000, )
#
#     test_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": np.array(X_test)},
#         y=np.array(y_test),
#         num_epochs=1,
#         shuffle=False)
#
#     accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
#
#     print("Accuracy score is %s with hidden layers %s" % (accuracy_score, i))
#
#     if accuracy_score > best_accuracy:
#         best_accuracy = accuracy_score
#         best_layers = i
#
# print("The best accuracy %s was trained with layers %s" % (best_accuracy, best_layers))


'''
The best accuracy 0.934141 was trained with layers [64, 128, 256], learning_rate=0.01, l2_regularization_strength=0.001

'''

#_______________________________________________________________________________________#
# Logistic Regression

# model_LogisticReg = LogisticRegression( penalty='l2', C=2,
#                                        solver="sag", max_iter=1000, n_jobs=-1)
# # tuned_parameters = {
# #                     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
# #                     'penalty':['l1','l2']
# #                     }
# #
# # LR = GridSearchCV(model_LogisticReg, tuned_parameters, cv=10)
# # LR.fit(X_train, y_train.values.ravel())
# # print(LR.best_params_) # l2, C=10
#
# model_LogisticReg.fit(X_train,y_train.values.ravel())
# y_pred = model_LogisticReg.predict(X_test)
# print(model_LogisticReg.score(X_test, y_test))

'''
l2, c=1, class_weight='balanced', score: 0.911591227391 ,
l2, c=1, score:0.93121415623
l2, c=100, score: 0.931214397035
l2, c=10, score: 0.931214637839
'''

#________________________________________________________________________________________#
# Decision Tree

