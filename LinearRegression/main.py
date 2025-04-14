from cProfile import label
from statistics import linear_regression
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


def load_and_normalize_data():
    # load the numpy arrays inputs and labels from the data folder
    X = np.loadtxt('./data/inputs.txt')
    y_l = np.loadtxt('./data/labels.txt')

    # normalize the target y
    y = y_l / np.std(y_l)

    return X, y


def data_summary(X, y):

    # return several statistics of the data

    X_mean = np.mean(X)
    X_std = np.std(X)
    y_mean = np.mean(y)
    y_std =np.std(y)
    X_min = np.min(X)
    X_max = np.max(X)

    return {'X_mean': X_mean,
            'X_std': X_std, 
            'X_min': X_min, 
            'X_max': X_max, 
            'y_mean': y_mean, 
            'y_std': y_std}


def data_split(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=4)

    return X_train, X_test, y_train, y_test, X_validation, y_validation


def fit_linear_regression(X, y, lmbda=0.0, regularization=None):
    """
    Fit a ridge regression model to the data, with regularization parameter lmbda and a given
    regularization method.
    If the selected regularization method is None, fit a linear regression model without a regularizer.

    !! Do not fit the intersept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    if regularization == None :
        linear_regression = LinearRegression(fit_intercept=False)
        linear_regression.fit(X, y)
        w = linear_regression.coef_
        c = linear_regression.intercept_

    if regularization == 'ridge' :
        linear_regression = Ridge(alpha=lmbda, fit_intercept=False)
        linear_regression.fit(X, y)
        w = linear_regression.coef_
        c = linear_regression.intercept_

    if regularization == 'lasso':
        linear_regression = Lasso(alpha=lmbda, fit_intercept=False)
        linear_regression.fit(X, y)
        w = linear_regression.coef_
        c = linear_regression.intercept_

    return w, c


def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features) data
    w: 1D numpy array of shape (n_features,) coefficients
    c: float intercept

    Returns: 1D numpy array of shape (n_samples,)
    """

    y_pred = X @ w + c
    return y_pred


def mse(y_pred, y):
    """
    Return the mean squared error between the predictions and the true labels.

    y_pred: 1D numpy array of shape (n_samples,)
    y: 1D numpy array of shape (n_samples,)

    Returns: float
    """

    MSE = mean_squared_error(y_pred, y)

    return MSE



def fit_predict_test(X_train, y_train, X_test, y_test, lmbda=0.0, regularization=None):
    """
    Fit a linear regression model, possibly with L2 regularization, to the training data.
    Record the training and testing MSEs.
    Use methods you wrote before

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbda: float, regularization parameter

    Returns: The coefficients and intercept of the fitted model, the training and testing MSEs in a dictionary.
    """

    w, c = fit_linear_regression(X_train, y_train, lmbda, regularization)

    y_train_pred = predict(X_train, w, c)
    y_test_pred = predict(X_test, w, c)

    mse_train = mse(y_train_pred, y_train)
    mse_test = mse(y_test_pred, y_test)


    results = {
        'mse_train': mse_train,
        'mse_test': mse_test,
        'lmbda': lmbda,
        'w': w,
        'c': c,
    }

    return results


def plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization=None, filename=None):
    """
    Plot the training and testing MSEs against the regularization parameter alpha.
    Use the functions you just wrote.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    alphas: list of values, the dataset percentage to be checked (alpha=n/d)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """
    
    mse_train_list = []
    mse_test_list = []

    for a in alphas :
        #n, d = X_train.shape
        n = int(a * d)
        results = fit_predict_test(X_train[:n, :], y_train[:n], X_test, y_test, lmbda, regularization)
        mse_train_list.append(results['mse_train'])
        mse_test_list.append(results['mse_test'])

    plt.plot(alphas, mse_train_list, label='train error')
    plt.plot(alphas, mse_test_list, label='test error')

    plt.xlabel('alpha=n/d')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()



def plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas,  plot_coefs, regularization='ridge', filename=None):
    """
    Plot the coefficients of the fitted model against the regularization parameter alpha.

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    lmbdas: list of values, the regularization parameter
    plot_coefs: list of integers, the coefficients of w to be plotted
    regularization: string, 'ridge' or 'lasso' or None
    filename: string, name to save the plot

    Returns: None
    """

    # TODO: You might want to use the pandas dataframe to store the results
    # your code goes here
    coefs = []
    for i in plot_coeffs :
        for lmbda in lmbdas :
            w, c = fit_linear_regression(X_train, y_train, lmbda, regularization)
            coefs.append(w[i])
        plt.plot(lmbdas, coefs, label = f'w{i}')   
        coefs = []

    plt.xlabel('lambda')
    plt.ylabel('weights')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()

def add_poly_features(X):
    """
    Add squared features to the data X and return a new vector X_poly that contains
    X_poly[i,j]   = X[i,j]
    X_poly[i,2*j] = X[i,j]^2

    X: 2D numpy array of shape (n_samples, n_features)

    Returns: 2D numpy array of shape (n_samples, 2 * n_features ) with the normal and squared features
    """

    # TODO
    #X_2 = X**2
    #X_poly = np.concatenate([X, X_2], axis=1)
    X_poly = np.concatenate((X, np.power(X,2)), axis= 1)

    return X_poly

def optimize_lambda(X_train, X_test, X_validation, y_train, y_test, y_validation, lmbdas, filename):
    """
    Optimize the regularization parameter lambda for the training data for ridge over the validation error and plot the validation error against the parameter lambda
    Show the best parameter lambda and the corresponding test error (not validation error!).

    X_train: 2D numpy array of shape (n_train_samples, n_features)
    y_train: 1D numpy array of shape (n_train_samples,)
    X_test: 2D numpy array of shape (n_test_samples, n_features)
    y_test: 1D numpy array of shape (n_test_samples,)
    X_validation: 2D numpy array of shape (n_validation_samples, n_features)
    y_validation: 1D numpy array of shape (n_validation_samples,)
    lmbdas: list of values, the regularization parameter, the bounds of the optimization range

    Returns: The best regularization parameter and the corresponding test error (!= validation error).
    """

    regularization='ridge'

    # TODO: Experiment code goes here
    mse_test_list = []
    mse_val_list = []


    for lmbda in lmbdas :
        results = fit_predict_test(X_train, y_train, X_test, y_test, lmbda, regularization)
        mse_test_list.append(results['mse_test'])

        results_val = fit_predict_test(X_train, y_train, X_validation, y_validation, lmbda, regularization)
        mse_val_list.append(results_val['mse_test'])


    best_test_mse = mse_test_list[np.argmin(mse_val_list)]
    best_lmbda = lmbdas[np.argmin(mse_val_list)]

    # TODO: Plotting code goes here

    plt.plot(lmbdas, mse_val_list, label='validation error')
    plt.xlabel ('regularization strenght (lambda)')

    plt.axvline(best_lmbda, color='red', label='Best Lambda')
    plt.legend()
    plt.savefig(f'results/{filename}.png')
    plt.clf()

    return best_test_mse, best_lmbda
  

if __name__ == "__main__":

    """ 
    !!!!! DO NOT CHANGE THE NAME OF THE PLOT FILES !!!!!
    They need to show in the Readme.md when you submit your code.

    It is executed when you run the script from the command line.
    'conda activate ml4phys-a1'
    'python main.py'

    This already includes the code for generating all the relevant plots.
    You need to fill in the ...
    """

    ## Exercise 1.
    # Load the data
    X, y = load_and_normalize_data()
    print("Successfully loaded and normalized data.")
    print(data_summary(X, y))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, X_validation, y_validation = data_split(X, y)

    n, d = X_train.shape

    ## Exercise 4.
    # Plot the learning curves
    print('Plotting dataset size vs. mse curve ...')
    alphas = np.linspace(2, 6, 1000) 
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda=0.0, regularization='ridge', filename='dataset_size_vs_mse')

    print('Plotting dataset size vs. mse curve for L2 ...')
    alphas = np.linspace(0.1, 2.5, 1000)
    lmbda = 0.001
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=001')

    lmbda = 10.0
    plot_dataset_size_vs_mse(X_train, y_train, X_test, y_test, alphas, lmbda, regularization='ridge', filename='dataset_size_vs_mse_l2=10')

    ## Exercise 5.
    print('Plotting regularizer vs. coefficient curve...')
    lmbdas = np.linspace(0,1000,200)
    plot_coeffs = [0, 3, 7, 8]
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas, plot_coeffs,regularization='ridge', filename='regularizer_vs_coefficients_Ridge')
    

    ## Exercise 6.
    print('Find the optimal parameters for the Ridge regression...')
    lmbdas = np.linspace(0.01, 100, 1000)
    n_ = 50
    lmbda, gen_error = optimize_lambda(X_train[:n_, :], X_test, X_validation, y_train[:n_], y_test, y_validation, lmbdas, filename='optimal_lambda_ridge_n50')
    print(n_, lmbda, gen_error)
    n_ = 150
    lmbda, gen_error = optimize_lambda(X_train[:n_, :], X_test, X_validation, y_train[:n_], y_test, y_validation, lmbdas, filename='optimal_lambda_ridge_n150')
    print(n_, lmbda, gen_error)
    print()

    ## Exercise 7.
    X_train_poly = add_poly_features(X_train) 
    X_test_poly = add_poly_features(X_test) 

    lmbdas = np.linspace(0.0001, 1.2, 1000 )
    plot_coeffs = [0, 2, 6, 7]
    plot_regularizer_vs_coefficients(X_train, y_train, X_test, y_test, lmbdas, plot_coeffs, regularization='lasso', filename='regularizer_vs_coefficients_LASSO')
    
    plot_coeffs = [0, 2, 6, 7, d+29, d+54]
    plot_regularizer_vs_coefficients(X_train_poly, y_train, X_test_poly, y_test, lmbdas, plot_coeffs, regularization='lasso', filename='regularizer_vs_coefficients_LASSO_polyfeat')

    print('Done. All results saved in the results folder.')

