---
title: "Scipy-notes"
author: "Aishwarya"
date: 2020-08-10
description: "-"
type: technical_note
draft: false
---

```python
import numpy as np
# 1. Find cubic root of 27 & 64 using cbrt() function
from scipy.special import cbrt
cb = cbrt([27, 64])
#print value of cb
print(cb) 
```

    [3. 4.]



```python
# 2.find permutation of 5, 2 using perm (N, k) function
from scipy.special import perm
per = perm(5, 2, exact = True)
print(per) 
```

    20



```python
# 3. Inverse Matrix
from scipy import linalg
# define square matrix
two_d_array = np.array([ [4,5], [3,2] ])
#pass value to function inv()
linalg.inv( two_d_array ) 

```




    array([[-0.28571429,  0.71428571],
           [ 0.42857143, -0.57142857]])




```python
#  4. Fourier transformation
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np 

#Frequency in terms of Hertz
fre  = 5 
#Sample rate
fre_samp = 50
t = np.linspace(0, 2, 2 * fre_samp, endpoint = False )
a = np.sin(fre  * 2 * np.pi * t)
figure, axis = plt.subplots()
axis.plot(t, a)
axis.set_xlabel ('Time (s)')
axis.set_ylabel ('Signal amplitude')
plt.show()
```


![png](scipy-notes_4_0.png)



```python
# 5. Eigen values and Eigen Vectors
arr = np.array([[5,4],[6,3]])
#pass value into function
eg_val, eg_vect = linalg.eig(arr)
#get eigenvalues
print(eg_val)
#get eigenvectors
print(eg_vect)
```

    [ 9.+0.j -1.+0.j]
    [[ 0.70710678 -0.5547002 ]
     [ 0.70710678  0.83205029]]



```python
# 6. Exponential Function
from scipy.special import exp10
exp = exp10([1,10])
print(exp)
```

    [1.e+01 1.e+10]



```python
# 7. Image manipyulation with scipy
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
#get face image of panda from misc package
panda = misc.face()
#plot or show image of face
plt.imshow( panda )
plt.show()

```


![png](scipy-notes_7_0.png)



```python
# 8. Rotation of image 
from scipy import ndimage, misc
from matplotlib import pyplot as plt
panda = misc.face()
#rotatation function of scipy for image – image rotated 127 degree
panda_rotate = ndimage.rotate(panda, 127)
plt.imshow(panda_rotate)
plt.show()
```


![png](scipy-notes_8_0.png)



```python
# 9.integration
from scipy import integrate

# Using quad as we can see in list quad is used for simple integration
# arg1: A lambda function which returns x squared for every x
# arg2: lower limit
# arg3: upper limit
result = integrate.quad(lambda x: x**2, 0, 3)
print(result) 
```

    (9.000000000000002, 9.992007221626411e-14)



```python
# 10. Solving Linear Algebra
import numpy as np
from scipy import linalg

# We are trying to solve a linear algebra system which can be given as:
# 1x + 2y =5
# 3x + 4y =6

# Create input array
A= np.array([[1,2],[3,4]])

# Solution Array
B= np.array([[5],[6]])

# Solve the linear algebra
X= linalg.solve(A,B)

# Print results
print(X)

# Checking Results
print("\n Checking results, following vector should be all zeros")
print(A.dot(X)-B) 
```

    [[-4. ]
     [ 4.5]]
    
     Checking results, following vector should be all zeros
    [[0.]
     [0.]]



```python
# 11. csr_matrix
import numpy as np
G_dense = np.array([ [0, 2, 1],
                     [2, 0, 0],
                     [1, 0, 0] ])
G_masked = np.ma.masked_values(G_dense, 0)
from scipy.sparse import csr_matrix
G_sparse = csr_matrix(G_dense)
print( G_sparse.data)
```

    [2 1 2 1]



```python
# 12. Dijikstra's Algorithm

from scipy.sparse.csgraph import dijkstra
graph = [

[0, 1, 2, 0],

[0, 0, 0, 1],

[0, 0, 0, 3],

[0, 0, 0, 0]

]

graph = csr_matrix(graph)
dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)

print(dist_matrix)
print(predecessors)
```

    [0. 1. 2. 2.]
    [-9999     0     0     1]



```python
# 13.Nelder–Mead Simplex Algorithm
import numpy as np

from scipy.optimize import minimize

def rosen(x):

    """The Rosenbrock function"""

    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = minimize(rosen, x0, method='nelder-mead',

options={'xatol': 1e-8, 'disp': True})

print(res.x)
```

    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 339
             Function evaluations: 571
    [1. 1. 1. 1. 1.]



```python
#14. Delanay triangle
from scipy.spatial import Delaunay
points = np.array([[0, 4], [2, 1.1], [1, 3], [1, 2]])
tri = Delaunay(points)
import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
```


![png](scipy-notes_14_0.png)



```python
#15. Convex Hull
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
points = np.random.rand(10, 2) # 30 random points in 2-D
hull = ConvexHull(points)
import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.show()
```


![png](scipy-notes_15_0.png)



```python
# 16. Import pi constant from both the packages
from scipy.constants import pi
from math import pi as p
print("sciPy - pi = %.16f",pi)
print("math - pi = %.16f",p)
```

    sciPy - pi = %.16f 3.141592653589793
    math - pi = %.16f 3.141592653589793



```python
#17. Scipy constatnts
import scipy.constants
res = scipy.constants.physical_constants["alpha particle mass"]
print (res)
```

    (6.6446573357e-27, 'kg', 2e-36)



```python
# 18.Orthogonal Distance Regression,
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import random
# Initiate some data, giving some randomness using random.random().
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([i**2 + random.random() for i in x])
# Define a function (quadratic in our case) to fit the data with.
def linear_func(p, x):
   m, c = p
   return m*x + c
# Create a model for fitting.
linear_model = Model(linear_func)
# Create a RealData object using our initiated data from above.
data = RealData(x, y)
# Set up ODR with the model and data.
odr = ODR(data, linear_model, beta0=[0., 1.])
# Run the regression.
out = odr.run()
# Use the in-built pprint method to give us results.
out.pprint()

```

    Beta: [ 5.42595108 -4.14111339]
    Beta Std Error: [0.77967664 2.33350109]
    Beta Covariance: [[ 1.87386452 -4.68466183]
     [-4.68466183 16.78514723]]
    Residual Variance: 0.32440748047361717
    Inverse Condition #: 0.1464266269832005
    Reason(s) for Halting:
      Sum of squares convergence



```python
# 19. Save mat file and loading
import scipy.io as sio
import numpy as np
#Save a mat file
vect = np.arange(10)
sio.savemat('array.mat', {'vect':vect})
#Now Load the File
mat_file_content = sio.loadmat('array.mat')
print (mat_file_content)
```

    {'__header__': b'MATLAB 5.0 MAT-file Platform: posix, Created on: Thu Aug  6 17:05:31 2020', '__version__': '1.0', '__globals__': [], 'vect': array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])}



```python
# 20.Interpolate
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)
print (x,y)
```

    [0.         0.36363636 0.72727273 1.09090909 1.45454545 1.81818182
     2.18181818 2.54545455 2.90909091 3.27272727 3.63636364 4.        ] [-0.65364362 -0.61966189 -0.51077021 -0.31047698 -0.00715476  0.37976236
      0.76715099  0.99239518  0.85886263  0.27994201 -0.52586509 -0.99582185]



```python
# 21. Scipy linalg
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np
#Declaring the numpy arrays
a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])
#Passing the values to the solve function
x = linalg.solve(a, b)
#printing the result array
print (x)
```

    [ 2. -2.  9.]



```python
# 22. Integrate
import scipy.integrate
from numpy import exp
from math import sqrt
f = lambda x, y : 16*x*y
g = lambda x : 0
h = lambda y : sqrt(1-4*y**2)
i = scipy.integrate.dblquad(f, 0, 0.5, g, h)
print (i)
```

    (0.5, 1.7092350012594845e-14)



```python
# 23. Determinant
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np
#Declaring the numpy array
A = np.array([[1,2],[3,4]])
#Passing the values to the det function
x = linalg.det(A)
#printing the result
print (x)
```

    -2.0



```python
#  24.  Univariate SPline
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)
plt.plot(x, y, 'ro', ms = 5)
plt.show()
```


![png](scipy-notes_24_0.png)



```python
# 25. Stats in Scipy- CDF
from scipy.stats import norm
import numpy as np
print (norm.cdf(np.array([1,-1., 0, 1, 3, 4, -2, 6])))
```

    [0.84134475 0.15865525 0.5        0.84134475 0.9986501  0.99996833
     0.02275013 1.        ]



```python

```


```python
import numpy as np
# 1. Find cubic root of 27 & 64 using cbrt() function
from scipy.special import cbrt
cb = cbrt([27, 64])
#print value of cb
print(cb) 

# 2.find permutation of 5, 2 using perm (N, k) function
from scipy.special import perm
per = perm(5, 2, exact = True)
print(per) 

# 3. Inverse Matrix
from scipy import linalg
# define square matrix
two_d_array = np.array([ [4,5], [3,2] ])
#pass value to function inv()
linalg.inv( two_d_array ) 


#  4. Fourier transformation
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np 

#Frequency in terms of Hertz
fre  = 5 
#Sample rate
fre_samp = 50
t = np.linspace(0, 2, 2 * fre_samp, endpoint = False )
a = np.sin(fre  * 2 * np.pi * t)
figure, axis = plt.subplots()
axis.plot(t, a)
axis.set_xlabel ('Time (s)')
axis.set_ylabel ('Signal amplitude')
plt.show()

# 5. Eigen values and Eigen Vectors
arr = np.array([[5,4],[6,3]])
#pass value into function
eg_val, eg_vect = linalg.eig(arr)
#get eigenvalues
print(eg_val)
#get eigenvectors
print(eg_vect)

# 6. Exponential Function
from scipy.special import exp10
exp = exp10([1,10])
print(exp)

# 7. Image manipyulation with scipy
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
#get face image of panda from misc package
panda = misc.face()
#plot or show image of face
plt.imshow( panda )
plt.show()


# 8. Rotation of image 
from scipy import ndimage, misc
from matplotlib import pyplot as plt
panda = misc.face()
#rotatation function of scipy for image – image rotated 127 degree
panda_rotate = ndimage.rotate(panda, 127)
plt.imshow(panda_rotate)
plt.show()

# 9.integration
from scipy import integrate

# Using quad as we can see in list quad is used for simple integration
# arg1: A lambda function which returns x squared for every x
# arg2: lower limit
# arg3: upper limit
result = integrate.quad(lambda x: x**2, 0, 3)
print(result) 

# 10. Solving Linear Algebra
import numpy as np
from scipy import linalg

# We are trying to solve a linear algebra system which can be given as:
# 1x + 2y =5
# 3x + 4y =6

# Create input array
A= np.array([[1,2],[3,4]])

# Solution Array
B= np.array([[5],[6]])

# Solve the linear algebra
X= linalg.solve(A,B)

# Print results
print(X)

# Checking Results
print("\n Checking results, following vector should be all zeros")
print(A.dot(X)-B) 

# 11. csr_matrix
import numpy as np
G_dense = np.array([ [0, 2, 1],
                     [2, 0, 0],
                     [1, 0, 0] ])
G_masked = np.ma.masked_values(G_dense, 0)
from scipy.sparse import csr_matrix
G_sparse = csr_matrix(G_dense)
print( G_sparse.data)

# 12. Dijikstra's Algorithm

from scipy.sparse.csgraph import dijkstra
graph = [

[0, 1, 2, 0],

[0, 0, 0, 1],

[0, 0, 0, 3],

[0, 0, 0, 0]

]

graph = csr_matrix(graph)
dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)

print(dist_matrix)
print(predecessors)

# 13.Nelder–Mead Simplex Algorithm
import numpy as np

from scipy.optimize import minimize

def rosen(x):

    """The Rosenbrock function"""

    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = minimize(rosen, x0, method='nelder-mead',

options={'xatol': 1e-8, 'disp': True})

print(res.x)

#14. Delanay triangle
from scipy.spatial import Delaunay
points = np.array([[0, 4], [2, 1.1], [1, 3], [1, 2]])
tri = Delaunay(points)
import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()

#15. Convex Hull
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
points = np.random.rand(10, 2) # 30 random points in 2-D
hull = ConvexHull(points)
import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.show()

# 16. Import pi constant from both the packages
from scipy.constants import pi
from math import pi as p
print("sciPy - pi = %.16f",pi)
print("math - pi = %.16f",p)

#17. Scipy constatnts
import scipy.constants
res = scipy.constants.physical_constants["alpha particle mass"]
print (res)

# 18.Orthogonal Distance Regression,
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import random
# Initiate some data, giving some randomness using random.random().
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([i**2 + random.random() for i in x])
# Define a function (quadratic in our case) to fit the data with.
def linear_func(p, x):
   m, c = p
   return m*x + c
# Create a model for fitting.
linear_model = Model(linear_func)
# Create a RealData object using our initiated data from above.
data = RealData(x, y)
# Set up ODR with the model and data.
odr = ODR(data, linear_model, beta0=[0., 1.])
# Run the regression.
out = odr.run()
# Use the in-built pprint method to give us results.
out.pprint()


# 19. Save mat file and loading
import scipy.io as sio
import numpy as np
#Save a mat file
vect = np.arange(10)
sio.savemat('array.mat', {'vect':vect})
#Now Load the File
mat_file_content = sio.loadmat('array.mat')
print (mat_file_content)

# 20.Interpolate
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)
print (x,y)

# 21. Scipy linalg
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np
#Declaring the numpy arrays
a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])
#Passing the values to the solve function
x = linalg.solve(a, b)
#printing the result array
print (x)

# 22. Integrate
import scipy.integrate
from numpy import exp
from math import sqrt
f = lambda x, y : 16*x*y
g = lambda x : 0
h = lambda y : sqrt(1-4*y**2)
i = scipy.integrate.dblquad(f, 0, 0.5, g, h)
print (i)

# 23. Determinant
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np
#Declaring the numpy array
A = np.array([[1,2],[3,4]])
#Passing the values to the det function
x = linalg.det(A)
#printing the result
print (x)

#  24.  Univariate SPline
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)
plt.plot(x, y, 'ro', ms = 5)
plt.show()

# 25. Stats in Scipy- CDF
from scipy.stats import norm
import numpy as np
print (norm.cdf(np.array([1,-1., 0, 1, 3, 4, -2, 6])))


```
