#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as np
import matplotlib.pyplot as plt


# In[156]:


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


# In[223]:




class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        print("outputgrad",output_gradient,"\n\n\n")
        print("self.input derivate",self.activation_prime(self.input),"\n\n\n")
        return output_gradient.T* self.activation_prime(self.input)


# In[224]:


#solo relu è fatta appositamente per adaw

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
        
class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.sum(np.maximum(0,x),axis=1)

        def relu_prime(x):
            drelu = np.zeros_like(x)
            drelu[x > 0] = 1
            return drelu

        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)


# In[225]:





# In[226]:



#ADAW NEURAL NETWORK (x2*x1.T).T

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, input_size)
        
    def forward(self, input):
        self.input = np.array(input)
        return self.weights*self.input.T + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = output_gradient* self.input.T
        input_gradient = self.weights*output_gradient
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


# In[227]:



def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


# In[230]:



def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
        print(output)
    print("\n");
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    print(f"{network[0].weights} \n\n{network[0].bias}\n\n")
    print(f"{network[2].weights} \n\n{network[2].bias}\n\n")
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                print(grad," \n\n")
                grad = layer.backward(grad, learning_rate)
                

        error /= len(x_train)
        if 1:#e>epochs-3:
            print(f"{e + 1}/{epochs}, error={error}")


# In[232]:



X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Relu(),
    Dense(3, 1),
    Relu()
]

# train
train(network, mse, mse_prime, X, Y, epochs=1, learning_rate=0.1)
# for x, y in zip(X, Y):
#     print(f"{network[0].weights} \n\n{network[0].bias}\n\n")
#     output = predict(network, x)
    
# decision boundary plot
# points = []
# for x in np.linspace(0, 1, 20):
#     for y in np.linspace(0, 1, 20):
#         z = predict(network, [[x], [y]])
#         points.append([x, y, z[0,0]])

# points = np.array(points)
# print(network[0].weights,network[0].bias,network[2].weights,network[2].bias)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
# plt.show()


# (self.weights.T*output_gradient).T



# In[200]:


x1 = np.arange(6.0).reshape((3,2 ))
x1


# In[202]:


x2 = np.arange(3.0).reshape((3,1))
x2


# In[131]:


(x2*x1.T).T


# In[203]:


x2*x1


# In[ ]:




