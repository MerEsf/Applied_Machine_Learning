

#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. 
# Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction.  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 
# 0 corresponds to an instance of not fraud.

import numpy as np
import pandas as pd

#  Calculate the percentage of the observations in the dataset that are instances of fraud

def answer_one():
    
    df = pd.read_csv('fraud_data.csv')
    ans = sum(df['Class']) / len(df['Class'])
    
    return ans


from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# train a dummy classifier that classifies everything as the majority class of the training data. 
# Calculate the accuracy of this classifier and the recall?

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score
    
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_majority.predict(X_test)
    accuracy_score = accuracy_score(y_test, y_dummy_predictions)
    recall_score = recall_score(y_test, y_dummy_predictions)
    
    return (accuracy_score, recall_score)

#  train a SVC classifer using the default parameters
# calculate the accuracy, recall, and precision


def answer_three():
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    predictions = svm.predict(X_test)
    accuracy_score = accuracy_score(y_test, predictions)
    recall_score = recall_score(y_test, predictions)
    precision_score = precision_score(y_test, predictions)
    
    return (accuracy_score, recall_score, precision_score)

# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, 
# Find the confusion matrix when using a threshold of -220 on the decision function. 

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svm = SVC(C= 1e9, gamma = 1e-07).fit(X_train, y_train)
    svm_predicted = svm.decision_function(X_test) > -220
    confusion_matrix = confusion_matrix(y_test, svm_predicted)
    
    return confusion_matrix

# Train a logisitic regression classifier with default parameters 

# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?


def answer_five():

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    log_reg = LogisticRegression().fit(X_train, y_train)
    y_scores_log_reg = log_reg.fit(X_train, y_train).decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_log_reg)
    
    closest_zero = np.argmin(np.abs(precision-0.75))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    y_score_log_reg = log_reg.fit(X_train, y_train).decision_function(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_log_reg)
    roc_auc_log_reg = auc(fpr_lr, tpr_lr)

    closest_zero_fpr_lr = np.argmin(np.abs(fpr_lr - 0.16))
    closest_zero_tpr_lr = recall[closest_zero_fpr_lr]


    ans = (closest_zero_r, closest_zero_tpr_lr)    
    return ans

answer_five()

# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    log_reg = LogisticRegression()
    grid_values = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    grid_lr = GridSearchCV(log_reg, param_grid = grid_values, scoring = 'recall', cv=3)
    grid_lr.fit(X_train, y_train)
    ans = np.array(grid_lr.cv_results_['mean_test_score'].reshape(5,2))
    
    return ans

#def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);


