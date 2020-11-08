import numpy,random,math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# #generate data
numpy.random.seed(100)
classA=numpy.concatenate(
    (numpy.random.randn(10,2)*0.2+[1.5,0.5],
    numpy.random.randn(10,2)*0.2+[-1.5,0.5],  #-0.5?
))
classB=numpy.random.randn(20,2)*0.2 +[0.0,-0.5]

inputs=numpy.concatenate((classA,classB))
targets=numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))

N= inputs.shape[0] # number of samples

permute=list(range(N))
numpy.random.shuffle(permute)
inputs=inputs[permute,:]
targets=targets[permute]

#print(inputs)
#print(targets)

#construct SVM
# N=4
# inputs=[(1,2),(3,4.5),(1,-5),(5,4)]
# targets=[1,1,-1,-1]
def kernel(X,Y,methods):
    if methods=='linear':
        return numpy.dot(X,Y)
    if methods=='p2':
        return (numpy.dot(X,Y)+1)**2
    if methods=='rbf':
        return numpy.exp( numpy.dot((X-Y),(X-Y))/(-2*0.5**2) )

def pre_compute(T,X):
    l=N
    P=numpy.zeros((l,l))
    for i in range(l):
        for j in range(l):
            P[i][j]=T[i]*T[j]*kernel(X[i],X[j],'p2')
    print('call pre_compute')
    return P

P=pre_compute(targets,inputs) #global variance
#print(P)
def objective(A):
    firstpart=0             #not matrix way
    for i in range(N):
        for j in range(N):
            firstpart+=A[i]*A[j]*P[i,j]
    re=0.5*firstpart-numpy.sum(A)
    # print(f'A={A},result={re}')
    # print(re)
    #matrix way
    # re=numpy.dot(A,P)
    # re=numpy.dot(A,re)
    # re*=0.5
    # re-=numpy.sum(A)
    return re

def zerofun(A):
    re =numpy.dot(A,targets)
    return re

def caculate_B(alpha):
    sv=[]
    for index,a in enumerate(alpha):
        if a<0.00001:
            alpha[index]=0
            continue
        if a>0:
            ai=a
            xi=inputs[index]
            ti=targets[index]
            sv.append((ai,xi,ti))

    # print(f'new A={alpha}')
    # print(sv)
    b=0
    ss=sv[0] # choose first support vector as basis
    for i in range(N):
        b+=alpha[i]*targets[i]*kernel(ss[1],inputs[i],'p2')
    b-=ss[2]
    return b,sv


def indicator(x,y):
   re=0
   b,sv=caculate_B(alpha)
   for point in sv:
       re+=point[0]*point[2]*kernel((x,y),point[1],'p2')
   return re-b

#plot----------------
def drawgraph():

    plt.plot([p[0] for p in classA],
            [p[1] for p in classA],
            'b.')
    plt.plot([p[0] for p in classB],
            [p[1] for p in classB],
            'r.')
    plt.axis('equal')

    xgrid=numpy.linspace (-5 , 5)
    ygrid=numpy.linspace (-4 , 4)

    grid=numpy.array ([[indicator(x,y)
                        for x in xgrid]
                        for y in ygrid] )
    plt.contour( xgrid , ygrid , grid ,
    ( -1.0 , 0.0 , 1.0 ) ,
    colors=( 'red' , 'black' , 'blue') ,
    linewidths =(1 , 3 , 1 ))

    plt.show()

def select(alpha):
    sv_alpha=[]
    for index,a in enumerate(alpha):
        if a>0.00001:
            sv_alpha.append(a)
            continue
    return sv_alpha
# main---------------
start=numpy.zeros(N)  # original A value
C=0.2   #slack variables
B=[(0,C) for b in range(N)]
XC={'type':'eq', 'fun':zerofun}

ret= minimize(objective,start,bounds=B,constraints=XC)
alpha=ret['x']
result=ret['success']
print(alpha,result)
sv_alpha=select(alpha)
print(f'sv_aplha={sv_alpha}')

drawgraph()








