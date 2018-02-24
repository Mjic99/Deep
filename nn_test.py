import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def genData(size):
    X = np.empty((3,size))
    Y = np.empty((1,size))
    one=1
    two=1
    for i in range(size):
        if i<size/2:
            X[0][i]=1
        else:
            X[0][i]=0
        
        if i%(size/4)==0:
            if one==0:
                one=1
            else:
                one=0
        X[1][i]=one
        
        if i%(size/8)==0:
            if two==0:
                two=1
            else:
                two=0
        X[2][i]=two
        
        if (X[0][i] or X[1][i]) and (X[1][i] != X[2][i]):
            Y[0][i]=1
        else:
            Y[0][i]=0   
        
    return X, Y

def forward_prop(X):
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
       
X, Y = genData(40)

n = X.shape[0]
m = X.shape[1]
h = 10           #hidden layer size
y = Y.shape[0]  #output size

W1 = np.random.randn(h,n)
b1 = np.zeros((h,1))
W2 = np.random.randn(1,h)
b2 = np.zeros((y,1))

iterations = 10000
alpha = 0.005

for i in range(iterations):
    
    #forward prop
    Z1, A1, Z2, A2 = forward_prop(X)
    
    #cost
    J = -(1/m)*np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2))
    
    #backprop
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    #update
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    

#taste
X_test, Y_test = genData(8)

A_test = forward_prop(X_test)[3]
print(X_test)
print(Y_test)
print(A_test)
res = np.around(A_test)
print(res)

    
    
    