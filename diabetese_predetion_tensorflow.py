import pandas as pd
# read the csv file of the dataset
diabetes = pd.read_csv('diabetes.csv')
print(diabetes.head())

# print(diabetes.columns)
# select columns to normalize i.e. get all the value between 0 and 1
cols_to_norm = ['Pregnancies','Glucose','BloodPressure',
                'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']
# normalization using lambda function
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# diabetes.head()



import tensorflow as tf

# passing all the feature columns as the numbric_columns to TensorFlo
preg = tf.feature_column.numeric_column('Pregnancies')
gluc = tf.feature_column.numeric_column('Glucose')
blood = tf.feature_column.numeric_column('BloodPressure')
skin = tf.feature_column.numeric_column('SkinThickness')
insu = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pred = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age = tf.feature_column.numeric_column('Age')


import matplotlib.pyplot as plt
# %matplotlib inline
# plot the histogram for column Age
diabetes['Age'].hist(bins = 20)
plt.show()


# converting age form numeric_column to bucketized_column
age_buckets = tf.feature_column.bucketized_column(age , [20,30,40,50,60,70,80])
# features columns
feat_cols = [preg ,gluc ,blood ,skin ,insu ,bmi ,pred ,age_buckets]

# forming the training and testing datasets
x_data = diabetes.drop('Outcome',axis=1)
labels = diabetes['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33,random_state=101)
# test_size = 0.33 refers to 33% of data will be the testing data
# random_state = 101 is used to repeat the same random data as in the previous iteration

# input function to be fed into the LinearClassifier model
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train , y = y_train
                                                 ,batch_size=10
                                                 ,num_epochs=1000
                                                 ,shuffle=True)
# defining the model
model = tf.estimator.LinearClassifier(feature_columns = feat_cols,n_classes=2)
# training the model on the input function
model.train(input_fn=input_func,steps=1000)


# predicton function which will only take X_test data set and when we use .predict() function it will guess the y_test dataset
pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x =X_test,
    batch_size = 10,
    num_epochs = 1,
    shuffle = False
    )
predictions = model.predict(pred_input_func)


# pritty print : pprint() is imported to print each parameter of the list in different lines
from pprint import pprint
pprint(list(predictions))

# evaluation input function to check for the accuracy and other parameters of the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test,
                                                     y = y_test,
                                                     batch_size=10,
                                                     num_epochs=1,
                                                     shuffle=False)
results = model.evaluate(eval_input_func)
pprint(results)

