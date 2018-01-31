import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#Load training data
data = pd.read_csv("Training_new_format_mean.csv",)

print("Training data loaded!")
data = data.drop(['xid', 'yid', 'hour', 'date_id', 'max', 'min', 'mean'], axis=1)

#_________________________________________________________________________________________#
#Prepare

X = data.iloc[:, :-1]
Y = (data.iloc[:, -1:]>=15).astype(int)

safe_points_num = (Y==False).sum()[0]
danger_points_num = (Y==True).sum()[0]

print("There are %s safe points in data." % safe_points_num)
print("There are %s dangerous points in data." % danger_points_num)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#____________________________________________________________________________________________#
# TensorFlow Neural Networks

feature_columns = [tf.feature_column.numeric_column("x", shape=[10])]

best_accuracy = 0
best_layers = -1
temp_dict = {}
temp_dict['predict'] = []

classifier = tf.estimator.DNNClassifier(
                                        feature_columns=feature_columns,
                                        hidden_units=[75, 150, 300],
                                        n_classes=2,
                                        optimizer=tf.train.ProximalAdagradOptimizer(
                                            learning_rate=0.01,
                                            l2_regularization_strength=0.0001),
                                        activation_fn=tf.nn.relu,
                                        )

train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": np.array(X_train)},
                    y=np.array(y_train),
                    num_epochs=None,
                    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(X_test)},
    y=np.array(y_test),
    num_epochs=1,
    shuffle=False)

classifier.train(input_fn=train_input_fn, steps=1000000,)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print(accuracy_score)

print("Classifier is trained!")

#_____________________________________________________________________________________#

#Load test data set
testing_data = pd.read_csv("Testing_replace_with_mean.csv",)


print("Testing data loaded!")
testing_data_drop = testing_data.drop(['xid', 'yid', 'hour', 'date_id', 'max', 'min', 'mean'], axis=1)


Test_X_input = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(testing_data_drop)},
                y=None,
                num_epochs=1,
                shuffle=False)

print("Prediction starts!")
predictions = classifier.predict(input_fn=Test_X_input, predict_keys='class_ids')

for j in predictions:
    temp_dict['predict'].append(int(j['class_ids']))

temp_df = pd.DataFrame(temp_dict)

result_df = pd.concat([testing_data, temp_df], axis=1)

print("Generating CSV file!")
# result_df.to_csv("NN_predict_0124.csv", index=False)
