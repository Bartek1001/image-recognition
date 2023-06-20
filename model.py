import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from random import randrange
import seaborn as sns
#import numpy as np

digits = pd.read_csv("train.csv")

print(digits.shape)
print(digits.head())

x = digits.drop(columns=['label'])
y = digits['label']

x2 = x.values.reshape(-1,28,28,1)
print(x2.shape)


index = randrange(4200)
plt.imshow(x2[index], cmap='gray')
plt.title("This image is: " + str(y[index]))
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 1984)
logistic = LogisticRegression()
logistic.fit(x_train, y_train)

logistic_predictions = logistic.predict(x_test)
logistic_accuracy = logistic.score(x_test,y_test)
logistic_accuracy = "{:.0%}".format(logistic_accuracy)
print("The accuracy of classifications using logistic regression is: " + str(logistic_accuracy))

x_test_2 = x_test.values.reshape(-1,28,28,1)
index2 = randrange(4200)
plt.imshow(x_test_2[index2], cmap='gray')
plt.title("This is the observation number: "+str(index2)+"\n"
          +"the actual label for this image is: "+str(y_test.iloc[index2])+"\n"
          +"the classification for this image is :"+str(logistic.predict(x_test)[index2]))
plt.show()

confusionmatrix = metrics.confusion_matrix(y_test,logistic_predictions)
plt.figure(figsize=(10,10))
sns.heatmap(confusionmatrix, annot=True, fmt='d')
plt.ylabel("Actual digits")
plt.xlabel("Recognized digits")
plt.show()
