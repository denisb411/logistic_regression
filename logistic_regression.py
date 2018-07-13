
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

df = pd.read_csv('ecommerce_data.csv')
df.head(20)


# In[3]:


df = pd.read_csv('ecommerce_data.csv')
data = df.as_matrix()


# In[4]:


X = data[:,:-1]
Y = data[:,-1]

X[:20]


# In[5]:


#normalizing columns 1 and 2
#Z = (X - mu)/std

X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

X[:20]


# In[6]:


#creating one hot encoding for the column 4
N, D = X.shape
X2 = np.zeros((N, D+3))
X2[:,0:(D-1)] = X[:,0:(D-1)]
for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1


# In[7]:


#another way of doing this
Z = np.zeros((N,4))
Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
X2[:,-4:] = Z

assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

Z


# In[8]:


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    
    X = data[:,:-1]
    Y = data[:,-1]
    
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
    
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
        
#     Z = np.zeros((N,4))
#     Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
#     X2[:,-4:] = Z
    
#     assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)
    
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


# In[9]:


#Binary classification as logistic just do binary
X, Y = get_binary_data()

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b) #This returns probabilities of Y given X
predictions = np.round(P_Y_given_X) #Round to make the prediction

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))


# There's no training until now

# ## Using Gradient Descent ##

# In[18]:


from sklearn.utils import shuffle

X, Y = get_binary_data()
X, Y = shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0


# In[20]:


from matplotlib import pyplot as plt

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(100000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    if i % 1000 ==0:
        print(i, ctrain, ctest)
        
print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))
      
legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()

