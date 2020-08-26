import keras
import keras.backend as K
import numpy as np
from keras.utils import to_categorical
from keras import layers
from keras import models, optimizers
from sklearn.utils import shuffle
import pandas as pd

def load_spam_data(split_ratio=0.6):
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',header=None).values
    data = shuffle(data,random_state=0)
    split = int(data.shape[0]*split_ratio)
    l = data.shape[1]
    X_train, y_train = data[:split,:(l-1)], data[:split,-1]
    X_test, y_test = data[split:,:(l-1)], data[split:,-1]
    
    val_split = int(X_train.shape[0]*split_ratio)
    #return X_train, y_train, , X_test, y_test
    return X_train[:val_split], y_train[:val_split],X_train[val_split:], y_train[val_split:], X_test, y_test

def impute_data(data,impute_thresh):
    ind = []
    impute = []
    for i in range(data.shape[1]):
        s = np.where(data[:,i]=='?')[0]
        if len(s) >= impute_thresh:
            ind.append(i)
        elif len(s) >= 1 and len(s) < impute_thresh:
            impute.append(i)
    data = pd.DataFrame(data).drop(columns=ind).values
    for i in impute:
        col = data[:,i]
        mean_col = col[np.where(col!='?')[0]].astype(float).mean()
        data[np.where(col=='?')[0],i] = mean_col
    return data.astype(float), ind, impute

def load_crime_data(split_ratio = 0.6, impute_thresh=20, header_file='headers.txt'):
    tmp = []
    headers = {}
    data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data',header=None).drop(columns=[0,1,2,3,4]).values
    with open('headers.txt','r') as f:
        for x in f:
            tmp.append(x.split(' ')[1])
    for i, h in enumerate(tmp[5:]):
        headers[i] = h
    data, ind, impute = impute_data(data,impute_thresh)
    split = int(data.shape[0]*split_ratio)
    l = data.shape[1]
    X_train, y_train = data[:split,:(l-1)], data[:split,-1]
    X_test, y_test = data[split:,:(l-1)], data[split:,-1]
    return X_train, y_train, X_test, y_test, headers, ind, impute

def prepare_data(load, split_ratio = 0.8, subset = 1):
    """
    load: the load_data() function from any given data module in keras
    split_ratio: ratio to split into train and validation
    Vectorize X, one hot encoding y (containing only 3 classes) + split into training and validation
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

class DNN():
    def __init__(self, input_dim, output_dim, depth, units,dropout=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.units = units
        self.dropout=dropout
        
    def build_model(self,hidden_act = 'relu',out_act='softmax',learning_rate=0.000001,
                   loss = "categorical_crossentropy",metrics = ['accuracy']):
        self.model = models.Sequential()
        self.model.add(layers.Dense(self.units, activation = hidden_act, input_shape=(self.input_dim,)))
        if self.dropout:
            self.model.add(layers.Dropout(0.4))
        for i in range(self.depth-1):
            self.model.add(layers.Dense(self.units, activation = hidden_act))
            if self.dropout:
                self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(self.output_dim,activation = out_act))
        self.model.compile(
                optimizer=optimizers.SGD(lr=learning_rate),
                loss = loss,
                metrics = metrics)
        
    def fit(self,inp,target,hyperparams, **kwargs):
        K.clear_session()
        if hyperparams:
            self.build_model(**hyperparams)
        else:
            self.build_model()
        self.history = self.model.fit(inp,target, **kwargs)
    
#     def predict(self,inp_test,*args,**kwargs):
#         return self.model.predict(inp_test, *args,**kwargs)