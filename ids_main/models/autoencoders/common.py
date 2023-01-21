import numpy as np

reconstruction_error_funcs = {
    'Euclidean Distance' : lambda x1,x2:np.sqrt(np.sum(np.square(x2-x1),axis=1)) ,
    'Gaussian RBF' : lambda x1,x2,e=1:1-np.exp(-e*np.sum(np.square(x2-x1),axis=1)) ,
}

