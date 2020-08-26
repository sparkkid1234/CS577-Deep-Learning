import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

""" HELPER FUNCTIONS"""

def xavier_init(fan_in, fan_out, constant=1):
    """
    fan_in: number of rows
    fan_out: number of columns
    return: a random matrix of shape (fan_in,fan_out) with specific characteristics
    purpose: initialize weights and biases
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low, high,(fan_in, fan_out))
def encode(val):
    """
    val: class in form of string
    return: 0, 1 or 2 depending on the class
    purpose: to help encode string classes into integers for iris dataset
    """
    if val == 'Iris-setosa': 
        return 0
    elif val == 'Iris-versicolor':
        return 1
    return 2
def load_iris(split_ratio=0.6):
    """
    return: training and testing data for iris dataset
    """
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data',header=None)
    data[4] = data[4].apply(lambda val: encode(val))
    data = data.values
    data = shuffle(data,random_state=0)
    split = int(data.shape[0]*split_ratio)
    l = data.shape[1]
    X_train, y_train = data[:split,:(l-1)], data[:split,-1]
    X_test, y_test = data[split:,:(l-1)], data[split:,-1]

    #val_split = int(X_train.shape[0]*split_ratio)
    #return X_train, y_train, , X_test, y_test
    return X_train,to_categorical(y_train), X_test, to_categorical(y_test)

def prepare_data(load, split_ratio = 0.8, subset = 1):
    """
    load: the load_data() function from any given data module in keras
    return: training, validating and testing data
    """
    (X_train, y_train), (X_test, y_test) = load()
    
    if subset:
        f = lambda y: np.where(np.logical_or(y==1,y==2,y==3))[0]
        tmp_train, tmp_test = y_train.flatten(), y_test.flatten()
        indice_train, indice_test = f(tmp_train), f(tmp_test)
        
    X_train, X_test = X_train.reshape((X_train.shape[0],3072))[indice_train], X_test.reshape((X_test.shape[0],3072))[indice_test]
    y_train, y_test = to_categorical(y_train[indice_train]), to_categorical(y_test[indice_test])
    split = int(X_train.shape[0]*split_ratio)
    return X_train[:split], y_train[:split], X_train[split:], y_train[split:], X_test, y_test

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    """
    x: mx3 input matrix (m is the number of instances, 3 is the number of classes)
    return: a mx3 matrix containing softmax value for each k=3 class for each example
    """
    res = np.exp(x).T/np.sum(np.exp(x),axis=1)
    return res.T

def categorical_cross_entropy(y,y_hat):
    """
    y: one-hot encoded target vector of shape mx3 (m is number of instances)
    y_hat: prediction/softmax output of shape mx3
    """
    return -(1/y.shape[0])*np.sum(y*np.log(y_hat))

def assert_dot(x,y):
    """
    assert if x dot y works
    """
    try:
        assert x.shape[1] == y.shape[0]
    except AssertionError:
        print("Can't take dot product due to inconsistency in x and y dimension")

""" GATES AND COMPUTATION GRAPH CLASS"""
"""
Each gate forward and backward computation has been designed carefully to be able to process a whole batch
of size m at once, instead of processing one instance at a time.
"""
class dotGate():
    def forward(self,x,w):
        assert_dot(x,w)
        h = x.dot(w)
        self.x = x
        self.w = w
        return h
    def backward(self,dldh):
        dldw = self.x.T.dot(dldh)
        dldx = dldh.dot(self.w.T)
        return dldw, dldx

class addGate():
    def forward(self,x,w):
        h = x+w
        self.x = x
        self.w = w
        return h
    def backward(self,dldh):
        #Sum across m examples to produce the vector of bias derivative
        return dldh, np.sum(dldh,axis=0).reshape(1,-1)
    
class sigmoidGate():
    def forward(self,x):
        h = sigmoid(x)
        self.x = x
        self.h = h
        return h
    def backward(self,dldh):
        dldx = dldh * self.h * (1-self.h)
        return dldx

class softmaxGate():
    def forward(self,x):
        h = softmax(x)
        self.x = x
        self.h = h
        return h
    def backward(self,dldh):
        dldx = []
        for i in range(len(self.h)): #For each of the m training examples
            s = self.h[i]
            softmax = s.reshape(-1,1)
            #Create a 3x3 softmax derivative matrix
            d = np.diagflat(s) - np.dot(softmax,softmax.T)
            #dldh * dhdx for this instance
            dldx.append(dldh[i].dot(d))
        dldx = np.array(dldx) #a mx3 matrix of derivative
        return dldx

class lossGate():
    def forward(self,y_hat,y):
        loss = categorical_cross_entropy(y,y_hat)
        self.y_hat = y_hat
        self.y = y
        return loss
    
    def backward(self):
        dldy_hat = -self.y/self.y_hat
        return dldy_hat

class ComputationGraph():
    def __init__(self):
        self.graph = [dotGate(),addGate(),sigmoidGate(),dotGate(),addGate(),sigmoidGate()
                      ,dotGate(),addGate(),softmaxGate(),lossGate()]
    def forward(self,X,y,W1,W2,W3,b1,b2,b3):
        inp = [X,W1,b1,W2,b2,W3,b3,y]
        res = []
        inp_ind = 0
        for gate in self.graph:
            g = type(gate).__name__
            #If this is the first dot gate
            if len(res) == 0:
                res.append(gate.forward(inp[0],inp[1]))
                inp_ind+=2
            else:
                if (g == 'addGate') or (g == 'dotGate') or (g == 'lossGate'):
                    res.append(gate.forward(res[-1],inp[inp_ind]))
                    inp_ind+=1
                else:
                    res.append(gate.forward(res[-1]))
        return res
    
    def backward(self):
        #Init gradient matrices/vectors of weights and biases
        grad = []
        bias_grad = []
        res = 1 #dldl
        for gate in self.graph[::-1]:
            g = type(gate).__name__
            #If this is the start of the backward pass
            if len(grad) == 0:
                res = gate.backward()
                grad.append(res)
            else:
                if g == 'addGate':
                    res, dldb = gate.backward(res)
                    bias_grad.append(dldb)
                elif g == 'dotGate':
                    dldw, res = gate.backward(res)
                    grad.append(dldw)
                else:
                    res = gate.backward(res)
        return grad[1:], bias_grad
    
""" DNN CLASS FOR TRAINING MODEL"""
class DenseNeuralNetwork():
    def __init__(self,units):
        #Number of units per hidden layer
        self.units=units
        
    def next_batch(self,X, y, batchSize):
        """
        return: a batch X,y of size batchSize
        """
        for i in np.arange(0, X.shape[0], batchSize):
            yield (X[i:i + batchSize], y[i:i + batchSize])
            
    def accuracy(self,y,y_hat):
        """
        y: mx3
        y_hat:mx3
        return: accuracy between y and y_hat
        """
        y = np.argmax(y,axis=1)
        y_hat = np.argmax(y_hat,axis=1)
        return accuracy_score(y,y_hat)
    
    def fit(self,X,y,**kwargs):
        #Assure y is one-hot encoded first
        try:
            assert y.shape[1] >= 2
        except AssertionError:
            print("y needs to be one-hot encoded")
            
        #Init weights
        self.W1 = xavier_init(X.shape[1],self.units)
        self.b1 = xavier_init(1,self.units)
        self.W2 = xavier_init(self.units,self.units)
        self.b2 = xavier_init(1,self.units)
        self.W3 = xavier_init(self.units,y.shape[1])
        self.b3 = xavier_init(1,y.shape[1])
        
        #Init hyperparams
        try:
            val_data = kwargs['val_data']
        except KeyError:
            val_data = None
        lr = None
        epochs = kwargs['epochs']
        eps0 = kwargs['learning_rate']
        batch_size = kwargs['batch_size']
        eps_final = eps0/100
        it = kwargs['iterations']
        #eps: a list of decaying learning rate to use for training
        eps = [((1-(i+1)/it)*eps0 + ((i+1)/it)*eps_final) for i in range(it)]
        eps.extend([eps_final]*(epochs-it))
        c = ComputationGraph()
        
        #SGD
        epochLoss = []
        val_epoch_loss = []
        epoch_acc = []
        val_epoch_acc = []
        for i in range(epochs):
            loss = []
            acc = []
            lr = eps[i]
            X_tr, y_tr = shuffle(X, y, random_state=0)
            for batchX, batchY in self.next_batch(X_tr,y_tr,batch_size):
                #Forward and backward pass
                res = c.forward(batchX,batchY,self.W1,self.W2,self.W3,self.b1,self.b2,self.b3)
                grad, bias_grad = c.backward()
                
                #Store training history of a batch
                loss.append(res[-1])
                acc.append(self.accuracy(batchY,res[-2]))
                    
                #update bias
                m = X.shape[0]
                self.b3 = self.b3 - lr*bias_grad[0]
                self.b2 = self.b2 - lr*bias_grad[1]
                self.b1 = self.b1 - lr*bias_grad[2]
                
                #Update weights
                self.W3 = self.W3 - lr*grad[0]
                self.W2 = self.W2 - lr*grad[1]
                self.W1 = self.W1 - lr*grad[2]
            #Average the loss and acc across all batches, that would be training history of an epoch
            epochLoss.append(np.average(loss))
            epoch_acc.append(np.average(acc))
            #If val_data is provided
            if val_data:
                c1 = ComputationGraph()
                X_val, y_val = val_data
                res_val = c1.forward(X_val,y_val,self.W1,self.W2,self.W3,self.b1,self.b2,self.b3)
                #Store validation history
                val_epoch_loss.append(res_val[-1])
                val_epoch_acc.append(self.accuracy(y_val,res_val[-2]))
        self.history = dict(loss=epochLoss,acc=epoch_acc,val_loss=val_epoch_loss,val_acc=val_epoch_acc)
    
    def evaluate(self,X,y):
        """
        return: accuracy and loss when evaluating dataset X and y using a trained model
        Needs to fit/train first to use this function
        """
        try:
            assert y.shape[1] >= 2
        except AssertionError:
            print("y needs to be one-hot encoded")
        c = ComputationGraph()
        res = c.forward(X,y,self.W1,self.W2,self.W3,self.b1,self.b2,self.b3)
        y_hat, loss = res[-2], res[-1]
        y = np.argmax(y,axis=1)
        y_hat = np.argmax(y_hat,axis=1)
        acc = accuracy_score(y,y_hat)
        return [acc, loss]