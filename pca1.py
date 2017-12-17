import numpy as np
import sys

np.set_printoptions(suppress=True)

def pca1(file):
    #Loading the iris data set
    Xt = np. genfromtxt(file, delimiter=',', autostrip=True)
    #print ("Xt=", Xt)

    #Loading the iris labels
    #y = np.genfromtxt('iris.labels.txt',delimiter=',', autostrip=True).astype("float")
    #print ("y=", y)

    X = Xt.T
    #print("X=", X)

    R = np.dot(X,Xt)
    #print("R=", R)

    evals, evecs = np.linalg.eigh(R)
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

    result = np.dot(Vt,X)
    #print ("Result = ", result)

    result_t = result.T
    return result_t

r = pca1(sys.argv[1])
np.savetxt(sys.argv[1]+'_pca1_output', r, delimiter=',')