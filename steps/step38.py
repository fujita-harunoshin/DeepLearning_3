if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.transpose(x)
y = x.T
y.backward()
print(x.grad)

A, B, C, D = 1, 2, 3, 4
x = Variable(np.random.randn(A, B, C, D))
print(x)
y = x.transpose(1, 0, 3, 2)
print(y)
y.backward()
print(x.grad)