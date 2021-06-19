### PROJEKT SI ###
### Bartosz Kubacki, Bartłomiej Urbanek ###
### Neuronowy model klienta bankowego (Bank Marketing Data) ###

## Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# 35 221 rows for learning
learn_dataset = pd.read_csv('bank_learn.csv', delimiter=';')
# 10 000 rows for testing
test_dataset = pd.read_csv('bank_test.csv', delimiter=';')

# Convert Categorical into codes
for col in learn_dataset.columns:
    if(learn_dataset.dtypes[col] == object):
        learn_dataset[col] = learn_dataset[col].astype('category').cat.codes
        
for col in test_dataset.columns:
    if(test_dataset.dtypes[col] == object):
        test_dataset[col] = test_dataset[col].astype('category').cat.codes


# Assign inputs and outputs
learn_x = learn_dataset.iloc[:,0:16]
learn_y = learn_dataset.iloc[:,16]

test_x = test_dataset.iloc[:,0:16]
test_y = test_dataset.iloc[:,16]

# Preprocessing
learn_x = preprocessing.StandardScaler().fit_transform(learn_x)
test_x = preprocessing.StandardScaler().fit_transform(test_x)

## Neural Network
from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier(random_state=1, 
                               tol=1e-07, 
                               alpha=1e-05, 
                               batch_size=60, 
                               activation='relu', 
                               solver='sgd', 
                               shuffle=True, 
                               hidden_layer_sizes=(30, 20, 10), max_iter=600)


neural_network.fit(learn_x, learn_y)

acc = accuracy_score(test_y, neural_network.predict(test_x))
print("Neural network model accuracy is {0:0.2f}".format(acc))



## Heatmap
import seaborn as sns;
conf_matrix_neural_network = confusion_matrix(test_y, neural_network.predict(test_x))
print('Confusion matrix:\n{}'.format(conf_matrix_neural_network))
sns.heatmap(conf_matrix_neural_network)

## PLot
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.plot(neural_network.loss_curve_)
plt.title('Neural network cost function loss')

plt.xlabel('epoch'); plt.ylabel('error value'); plt.grid();