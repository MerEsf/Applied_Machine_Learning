
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()

def answer_zero():
  
    return len(cancer['feature_names'])

answer_zero() 

def answer_one():
    
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    return df


answer_one()


# What is the class distribution? (i.e. how many instances of `malignant` (encoded 0) and how many `benign` (encoded 1)?)


def answer_two():
    cancerdf = answer_one()
    
    malignant = len(cancerdf[cancerdf['target'] == 0])
    benign = len(cancerdf[cancerdf['target'] == 1])
    ans =  pd.Series(data = [malignant, benign], index = ['malignant', 'benign'])
    
    return ans


answer_two()

def answer_three():
    cancerdf = answer_one()
    
    X = cancerdf.drop('target', axis = 1)
    y = cancerdf['target']
    
    return X, y


from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    return X_train, X_test, y_train, y_test


from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    ans = knn.fit(X_train, y_train)
    
    return ans

# Using your knn classifier, predict the class label using the mean value for each feature.


def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    ans = answer_five().predict(means)
    
    return ans


# Using your knn classifier, predict the class labels for the test set `X_test`.


def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    ans = knn.predict(X_test)
    
    return ans

# Find the score (mean accuracy) of your knn classifier using `X_test` and `y_test`.


def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    ans = knn.score(X_test,y_test)
    
    return ans

#  visualize the differet predicition scores between training and test sets, as well as malignant and benign cells.


def accuracy_plot():
    import matplotlib.pyplot as plt

    get_ipython().magic('matplotlib notebook')

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)


accuracy_plot() 




