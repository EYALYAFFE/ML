#%%
#1
import numpy as np
#%%
#2
print(np.__version__)

#%%
#3
x=np.zeros(10,dtype=int)
print(x)
#%%
#4
x1=np.zeros(10,dtype=int)
print("nbytes:", x1.nbytes, "bytes")

#%%
#5
?np.add
#%%
#6
x=np.zeros(10,int)
x[4]=1
print(x)
#%%
#7
x=np.arange(10,50)
print(x)
#%%
#8
x1 = np.random.randint(10, size=10)  # One-dimensional array
print(x1)
reversed_arr = x1[::-1]

print(reversed_arr)
#%%
#9
a = np.arange(9).reshape((3, 3))
print(a)
#%%
#10
a = np.array([1,2,0,0,4,0])
b=np.nonzero(a)
print(b) 
#%%
#11
a=np.eye(3, dtype=int)
print(a)
#%%
#12
#a=np.random.rand(3,3,3)
print(a)
#%%
#13
a=np.random.randint(10, size=(10, 10))
min_num=a.min()
max_num=a.max()
print(a)
print(min_num)
print(max_num)
#%%
#14
x1 = np.random.randint(10, size=30)
print(x1)
a_mean=x1.mean();
print(a_mean)
#%%
#15
a=(5,5)
b=np.ones(a)
print(b)
b[1:-1,1:-1]=0
print(b)
#%%
#16
A = np.array([1,2,3,4,5])
np.pad(A, (1, 1), 'constant')
#%%
#17
print(0 * np.nan)
print(np.nan == np.nan) 
print(np.inf > np.nan) 
print(np.nan - np.nan) 
print(0.3 == 3 * 0.1)
#%%
#18
diag=np.zeros((5,5),int)
print(diag)
np.fill_diagonal(diag, np.array([1,2,3,4,7]))
print(diag)
#%%
#19
b=np.ones((8,8),int)
b[1::2,::2] = 0
b[::2,1::2] = 0
print(b)
#%%
#20
print(np.unravel_index(100, (6,7,8)))
#%%
#21
z = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(z)
#%%
#22

#%%
#23
arr = np.arange(1,9,dtype=np.int16).reshape((2,4))
print(arr.dtype)
#%%
#24
mat1 = np.arange(1,16).reshape((5,3))
mat2 = np.arange(1,7).reshape((3,2))
print(mat1)
print(mat2)
mat3=np.dot(mat1,mat2)
print(mat3)
#%%
#25
Z=np.arange(11) 
Z[(3<Z)&(Z <= 8)] *= -1 
print(Z)
#%%
#26
"""print(sum(range(5),-1)) 
from numpy import * 
print(sum(range(5),-1))""" 
#%%
#27
Z=np.arange(11) 
print(Z)
z1=Z**Z
print(z1) 
z2=Z<-Z 
z3=1j*Z
#%%
#28
#np.array(0) / np.array(0) 
#np.array(0) // np.array(0) 
#%%
#29
#%%
#30
z=np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
print(z)
#%%
#31
#done
#%%
#32
#print(np.sqrt(-1)==np.emath.sqrt(-1)) 
#%%
#33
yesterday=np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print("Yestraday: ",yesterday)
today=np.datetime64('today', 'D')
print("Today: ",today)
tomorrow=np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Tomorrow: ",tomorrow)
#%%
#34
Z = np.arange('2016-08', '2016-09', dtype='datetime64[D]')
print(Z)
#%%
#35
A = np.array([1,2,3,4])
B = np.array([1,2,3,4])
print(A)
C=np.add(A, B)
print(C)
D=np.multiply(A,-0.5)
print(D)
print("************************")
E=np.multiply(C,D)
print(E)
#%%
#36
#36.1
import math
x = 1234.5678
y=math.modf(x) # (0.5678000000000338, 1234.0)
print(y)
#36.2
s = 1234.5678
i,d = divmod(s, 1)
print(i)
#%%
#37
import numpy as np
x = np.zeros((5,5))
print("Original array:")
print(x)
print("Row values ranging from 0 to 4.")
x += np.arange(5)
print(x)
#%%
#38-no exercise
#%%
#39
x = np.linspace(0,1,12,endpoint=True)[1:-1]
print(x)
#%%
#40
x = np.random.random(10)
print("Original array:")
print(x)
x.sort()
print("Sorted array:")
print(x)
####part 2####
#%%
#41
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
#%%
#42
np.set_printoptions(threshold=np.nan)
Z = np.zeros((25,25))
print(Z)
#%%
#43
Z = np.arange(100)
print(Z)
v = np.random.uniform(0,100)
print(v)
index = (np.abs(Z-v)).argmin()
print(Z[index])
#%%
#44
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(Z)
#%%
#45
Z = np.random.random((10,2))
print(Z)
X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
D = np.sqrt((X-X.T)**2 + (Y-Y.T)**2)
print(D)
#%%
#46
Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)
print(Z)
#%%
#47
#%%
#48
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
#%%
#49
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
#%%
#50
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
#%%
#51
a=np.arange(9).reshape((3, 3))
print(a)
b=a-a.mean(axis=1,keepdims=True)
print(b)
#%%
#52
Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
#%%
#53
Z = np.random.randint(0,3,(3,10))
print(Z)
print((~Z.any(axis=0)).any())
#%%
#54
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]
array = np.random.random(10)
print(array)
value=0.5
print(find_nearest(array, value))
#%%
#55
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
#%%
#56
Z=np.ones(10)
print(Z)
I = np.random.randint(0,len(Z),20)
print(I)
Z += np.bincount(I, minlength=len(Z))
print(Z)
#%%
#57
