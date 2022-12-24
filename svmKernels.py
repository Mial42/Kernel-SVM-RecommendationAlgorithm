"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 5 #2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    step1 = np.dot(X1, X2.T) + 1
    return step1 ** _polyDegree


def pairwiseSquaredDistance(X1, X2):
    """
        Arguments: 
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns: 
            The distance between X1 and X2 (the vector norm of X1 - X2), squared
    """
    n1 = len(X1)
    n2,d = X2.shape
    #Strategy: recall that (x - y)^2 = x^2 - 2xy + y^2 
    #Let x = X1 and y = X2

    #Square X1, then dot it with a 1's array of the same dimension as X2.T (x^2)
    temp1 = (X1**2).dot(np.ones((d,n2)))

    #Dot a 1's array of the same dimensions as X1 with X2.T squared (y^2)
    temp2 = np.ones((n1,d)).dot((X2.T**2))

    #Add -2 times X1 dot X2.T to the previous two results and return it (x^2 - 2xy + y^2)
    return  temp1 - 2*X1.dot(X2.T) + temp2

def myGaussianKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    norm = pairwiseSquaredDistance(X1,X2)
    kernel = np.exp((-norm/(2*_gaussSigma)))
    return kernel
    # norm = pairwiseSquaredDistance(X1,X2)
    # step1 = -1 * norm / (2 * _gaussSigma ** 2)
    # return np.exp(step1)
