import numpy as np
import sys

np.set_printoptions(suppress=True)

def scatter2(file, labels):
    #Loading the iris data set
    Xt = np. genfromtxt(file, delimiter=',', autostrip=True)
    #print ("Xt=", Xt)

    #Loading the iris labels
    y = np.genfromtxt(labels, delimiter=',', autostrip=True)
    #print ("y=", y)

    X = Xt.T
    #print ("X=", X)

    mu = np.mean(X, axis = 1)
    Xc = (X.T - mu).T
    M = np.dot(Xc, Xc.T)

    #print ("M=", M)

    m = Xt.shape[1]
    #print ("m=", m)
    W1 = np.zeros((m, m))
    W2 = np.zeros((m, m))
    W3 = np.zeros((m, m))
    s1 = np.zeros((1, m))
    s2 = np.zeros((1, m))
    s3 = np.zeros((1, m))
    m1 = 0
    m2 = 0
    m3 = 0

    for i, xt in enumerate(Xt):
        if y[i] == 1:
            #W1 += np.dot(xt[:,None], xt[None, :])
            s1 += xt
            m1 += 1
        elif y[i] == 2:
            #W2 += np.dot(xt[:,None], xt[None, :])
            s2 += xt
            m2 += 1
        elif y[i] == 3:
            #W3 += np.dot(xt[:,None], xt[None, :])
            s3 += xt
            m3 += 1

    #print ("m1 = ", m1)
    #print ("m2 = ", m2)
    #print ("m3 = ", m3)

    #print ("s1 =", s1)
    #print ("s2 =", s2)
    #print ("s3 =", s3)

    mu1 = s1 / m1
    mu2 = s2 / m2
    mu3 = s3 / m3

    #print ("mu1 = ", mu1)

    for i, xt in enumerate(Xt):
        if y[i] == 1:
            ct = xt - mu1
            W1 += np.dot(ct.T, ct)
        elif y[i] == 2:
            ct = xt - mu2
            W2 += np.dot(ct.T, ct)
        elif y[i] == 3:
            ct = xt - mu3
            W3 += np.dot(ct.T, ct)

    #print ("W1 = ", W1)
    #print ("W2 = ", W2)
    #print ("W3 = ", W3)

    W = W1+W2+W3
    #print ("W = ", W)

    B = M - W
    #print ("B = ", B)

    #Maximizing the between-class scatter
    evals, evecs = np.linalg.eigh(B)

    idx = np.argsort(evals)[:: -1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    r = 2
    V_r = evecs[:, :r]

    Vt = V_r.T

    result = np.dot(Vt, X)

    result_t = result.T
    return result_t

r = scatter2(sys.argv[1],sys.argv[2])
np.savetxt(sys.argv[1]+'_scatter2_output', r, delimiter=',')
