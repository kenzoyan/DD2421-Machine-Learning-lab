#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
    
    base=np.array([1 for i in range(Npts)]).reshape(-1,1)
    wx_sum=np.sum(base* W)

    #print(wx_sum)
    # TODO: compute the values of prior for each class!
    # ==========================
    for index,C in enumerate(classes):
        Ni= labels==C
        idx=np.where(labels==C)[0]
        #print(idx)
        a=len(idx)
        a=np.array([1 for i in range(a)]).reshape(-1,1)
        #b=W[idx,:]
        #print(a,b)
        Ni=np.sum(a*W[idx,:])
        #print(Ni)
        prior[index]=Ni/float(wx_sum)

    # ==========================
    #print(prior)
    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for jdx,C in enumerate(classes):
        idx= labels==C
        idx=np.where(labels==C)[0]
        xlc=X[idx,:]
        w=W[idx,:].reshape(-1)
        w_sum=np.sum(w)
        #print(f'w_sum={w_sum},wlen={len(idx)}')
        mu[jdx]=np.dot(w,xlc)/ float(w_sum)
        #print(mu[jdx],mu[jdx].shape)
        sigma[jdx]=np.diag(np.dot(w,(xlc-mu[jdx])**2)/float(w_sum))
    
    # ==========================
    # print(mu)
    # print("-------------")
    # print(sigma)
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    #print(f'Nclasses={Nclasses}')
    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    for j,xi in enumerate(X):
        for index in range(Nclasses):
            part1=np.log(np.linalg.det(sigma[index]))/(-2)
            #print("-----------")
            s1=(xi-mu[index])
            s2=np.transpose(xi-mu[index])
            #print(s1.shape,s2.shape)
            part2=(np.dot(np.dot(s1,(np.linalg.pinv(sigma[index]))),s2))/ -2
        
            #print((xi-mu[index]).shape,(np.linalg.pinv(sigma[index])).shape,(np.transpose(xi-mu[index])).shape)
            #print((xi-mu[index]),(np.linalg.pinv(sigma[index])),(np.transpose(xi-mu[index])))
            
            part3=np.log(prior[index])
            #print(f'Part1={part1}')
            #print(f'Part2={part2}')
            #print(f'Part3={part3}')
            logProb[index,j]= part1+part2+part3
    # ==========================
    #print(logProb)
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    #print(h)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.


#X, labels = genBlobs(centers=5)
# #W test
# W=[1/len(X) for i in range(len(X))]
# W=np.array(W).reshape(-1,1)
# #print(W)
# #print(X)
# pr=computePrior(labels)
# print(pr)
#mu, sigma = mlParams(X,labels)
# print(mu)
# print(sigma)
#plotGaussian(X,labels,mu,sigma)


# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BayesClassifier(), dataset='iris', split=0.7)
#plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
#plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)







# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))
        
        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        compare=labels==vote
        for index in range(len(compare)):
            if compare[index]==True:
                compare[index]=0
            elif compare[index]==False:
                compare[index]=1
        compare=compare.reshape(-1,1)
        # print(f'compare={compare}')
        # print(f'wCur={wCur}')
        #print('---------------')
        #print(aa,aa.shape)
        error=np.sum(wCur * compare)
        if error==0:
            error=0.5
          
        alpha=(np.log(1-error)-np.log(error))/2.0
        alphas.append(alpha) # you will need to append the new alpha
        #print(f'old w={wCur}')

        for index,c in enumerate(compare): #0==True; 1==False
            if c==0:
                #print(f'wCur[index]*np.exp(-alpha)={wCur[index],np.exp(-alpha)}') 
                wCur[index]=wCur[index]*np.exp(-alpha)
                #print(wCur[index])
            elif c==1:
                wCur[index]=wCur[index]*np.exp(alpha)
        #print(wCur)
        #normalization
        Z=np.sum(wCur)
        # print(f'old sum wcur Z={Z}')
        wCur=(wCur/Z).reshape(-1,1)
        # print(f'new sum wcur Z={np.sum(wCur)}')
        #print(f"T={i_iter}")
        # print(error)
        # print(alpha)
        #print(wCur)
        # ==========================
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for N in range(Nclasses):

            for i in range(len(alphas)):

                pred_class=classifiers[i].classify(X)
                compare=pred_class==N
                votes[:,N]+=np.dot(alphas[i],compare)
        
        
        #print(votes)

        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)
# plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)



# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
# plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel',split=0.7)


# Now repeat the steps with a decision tree classifier.


# testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



# testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



# plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



# plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


# testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
pca.fit(xTr) # use training data to fit the transform
xTrpca = pca.transform(xTr) # apply on training data
xTepca = pca.transform(xTe) # apply on test data
#use our pre-defined decision tree classifier together with the implemented
#boosting to classify data points in the training data

classifier1 = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr) #70.7
classifier2 = DecisionTreeClassifier().trainClassifier(xTrpca, yTr) #48.4
classifier3 = BayesClassifier().trainClassifier(xTrpca, yTr) # 87.7
classifier4 = BoostClassifier(DecisionTreeClassifier(), T=20).trainClassifier(xTrpca, yTr) # 76.1


#testClassifier(classifier2, dataset='olivetti',split=0.7,dim=20)
testClassifier(classifier1, dataset='olivetti',split=0.7,dim=20)
# classifier=classifier3
# yPr = classifier.classify(xTepca)
#choose a test point to visualize
# testind = random.randint(0, xTe.shape[0]-1)
#visualize the test point together with the training points used to train
#the class that the test point was classified to belong to
# visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

