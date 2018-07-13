
# coding: utf-8

# In[66]:


import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)


# In[67]:


w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)


# In[68]:


def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
cross_entropy(T, Y)


# In[69]:


#the solution weights
w = np.array([0,4,4])

z = Xb.dot(w)
Y = sigmoid(z)

cross_entropy(T, Y)


# In[70]:


w = np.random.randn(D + 1)
#finding solution with gradient descent
learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))
        
    # dJ/dwi = XT(Y - T)
    w += learning_rate * Xb.T.dot(T - Y)
    Y = sigmoid(Xb.dot(w))
    
print("Final w:", w)


# ## With L2 Regularization ##

# In[72]:


w = np.random.randn(D + 1)
#finding solution with gradient descent
learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))
        
    # dJ/dwi = XT(Y - T)
    w += learning_rate * (np.dot((T - Y).T, Xb) - 0.1*w)
    Y = sigmoid(Xb.dot(w))
    
print("Final w:", w)

