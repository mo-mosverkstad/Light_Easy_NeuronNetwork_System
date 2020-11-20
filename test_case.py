from lib.lenslib import Lens
import numpy as np

X = np.array( [ [-1,-1,-1,1],
                [-1,-1,1,1],
                [1,1,0,1],
                [1,1,1,1]] )
Y = np.array([[-1,-1,1,1]]).T

lens = Lens(X.shape)
print(lens.learn(X, Y))
print(lens.forward([0,0,1,-1]))