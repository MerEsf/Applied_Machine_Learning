
# ## Part 1 - Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5    #Returns evenly spaced numbers over a specified interval
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


def part1_scatter():
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib notebook')
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    
    

# Write a function that fits a polynomial LinearRegression model for degrees 1, 3, 6, and 9. 
# (Use PolynomialFeatures) 

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    degrees = [1, 3, 6, 9]
    ans = np.zeros((4,100))
    
    for i, d in enumerate(degrees): #keep a count of iterations
        poly = PolynomialFeatures(degree = d)
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        predict = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)))
        ans[i,:] = predict
    
    return ans

def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib notebook')
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)


# Write a function that fits a polynomial LinearRegression model for degrees 0 through 9. 
# For each model compute the $R^2$ (coefficient of determination) regression score on the training data as well as the the test data

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    
    for i in range(10):
        poly = PolynomialFeatures(degree = i)
        X_train_poly = poly.fit_transform(X_train.reshape(11,1))
        X_test_poly = poly.fit_transform(X_test.reshape(4,1))
        linreg = LinearRegression().fit(X_train_poly, y_train)
        r2_train[i] = linreg.score(X_train_poly, y_train)
        r2_test[i] = linreg.score(X_test_poly, y_test)

    return (r2_train, r2_test)



# Based on the $R^2$ scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? 
# What degree level corresponds to a model that is overfitting? 
# What choice of degree level would provide a model with good generalization performance on this dataset? 


def answer_three():
    import matplotlib.pyplot as plt    
    r2_train, r2_test = answer_two()
    degrees = np.arange(0, 10)
    plt.figure()
    plt.plot(degrees, r2_train, degrees, r2_test)
    
    return (3, 8, 7)


# train two models: a non-regularized LinearRegression model (default parameters) and 
# a regularized Lasso Regression model (with parameters `alpha=0.01`, `max_iter=10000`) both on polynomial features of degree 12. 


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    poly = PolynomialFeatures(12)
    X_poly = poly.fit_transform(X_train.reshape(11,1))
    X_test_poly = poly.fit_transform(X_test.reshape(4,1))
    linreg = LinearRegression().fit(X_poly, y_train)
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)
    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_poly, y_train)
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)
answer_four()


# ## Part 2 - Classification

# *Attribute Information:*
# 
# 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
# 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
# 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
# 4. bruises?: bruises=t, no=f 
# 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
# 6. gill-attachment: attached=a, descending=d, free=f, notched=n 
# 7. gill-spacing: close=c, crowded=w, distant=d 
# 8. gill-size: broad=b, narrow=n 
# 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
# 10. stalk-shape: enlarging=e, tapering=t 
# 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
# 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 16. veil-type: partial=p, universal=u 
# 17. veil-color: brown=n, orange=o, white=w, yellow=y 
# 18. ring-number: none=n, one=o, two=t 
# 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
# 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
# 21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
# 22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)


X_subset = X_test2
y_subset = y_test2

# train a DecisionTreeClassifier with default parameters and random_state=0. 
# What are the 5 most important features found by the decision tree?
# 
#  the feature names are available in the `X_train2.columns` property, 
# and the order of the features in `X_train2.columns` matches the order of the feature importance values 
# in the classifier's `feature_importances_` property. 

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    features = []
    # the feature names are available in the X_train2.columns property
    # the order of the features in X_train2.columns matches the order of the feature importance values in the classifier's feature_importances_ property
    for feature, importance in zip(X_train2.columns, clf.feature_importances_):
        features.append([importance, feature])
    features.sort(reverse = True)

    return [feature[1] for feature in features[:5]]
answer_five()


# 
# The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel.
# 
# explore the effect of `gamma` on classifier accuracy by using the `validation_curve` function 
# to find the training and test scores for 6 values of `gamma` from `0.0001` to `10` (i.e. `np.logspace(-4,1,6)`). 

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    
    svc = SVC(kernel = 'rbf', C = 1, random_state = 0)
    gamma = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(svc,X_subset,y_subset,
                                                 param_name='gamma',param_range=gamma,scoring='accuracy')
    
    train_scores = train_scores.mean(axis=1)
    test_scores = test_scores.mean(axis=1)

    return train_scores, test_scores

answer_six()

# what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? 
# What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? 
# What choice of gamma would be the best choice for a model with good generalization performance on this dataset 


def answer_seven():
    
train_scores, test_scores = answer_six()
gamma = np.logspace(-4,1,6)
plt.figure()
plt.plot(gamma, train_scores, 'b--', gamma, test_scores, 'g-')
    
    return (0.001, 10, 0.1)

