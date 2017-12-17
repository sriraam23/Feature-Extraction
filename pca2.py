import numpy as np
import sys

np.set_printoptions(suppress=True)

def pca2(file):
    #Loading the iris data set
    Xt = np. genfromtxt(file, delimiter=',', autostrip=True)
    #print ("Xt=", Xt)

    #Loading the iris labels
    #y = np.genfromtxt('iris.labels.txt',delimiter=',', autostrip=True).astype("float")
    #print ("y=", y)

    X = Xt.T
    #print("X=", X)

    mu = np.mean(X, axis=1)
    #print ("Mu=", mu)
    Xc = (X.T - mu).T
    C = np.dot(Xc, Xc.T)
    #print (" Xc =", Xc);
    #print ("C=", C)

    evals, evecs = np.linalg.eigh(C)
    #print("evals =", evals , "evecs =", evecs)

    idx = np.argsort( evals )[:: -1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    #print("evals =", evals , "evecs =", evecs)

    r = 2
    V_r = evecs[: ,:r]
    #print ("V_r=",V_r)

    Vt = V_r.T
    #print Vt

    result = np.dot(Vt,Xc)
    #print ("Result = ", result)

    result_t = result.T
    return result_t

r = pca2(sys.argv[1])
np.savetxt(sys.argv[1]+'_pca2_output', r, delimiter=',')