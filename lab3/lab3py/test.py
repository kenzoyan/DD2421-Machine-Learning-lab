import numpy as np

aa=np.array([5,4,3,2,1]).reshape(-1,1)
bb=np.array([0,0,0,0,0]).reshape(-1,1)
print(aa.shape,bb.shape)
print(aa*bb)
print(np.dot(aa,bb))

