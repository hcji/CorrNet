# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:14:27 2019

@author: hcji
"""

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Add, concatenate
from keras.engine.topology import Layer
from keras import optimizers
from scipy.stats import pearsonr

'''
data_l = np.load('Data/data_l.npy')
data_r = np.load('Data/data_r.npy')
label = np.load('Data/data_label.npy')

test_l = np.load('Data/test_v1.npy')
test_r = np.load('Data/test_v2.npy')
test_label = np.load('Data/test_l.npy')
'''
  
class ZeroPadding(Layer):
    def __init__(self, **kwargs):
        super(ZeroPadding, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.zeros_like(x)

    def get_output_shape_for(self, input_shape):
        return input_shape
    
    
class CorrnetCost(Layer):
    def __init__(self,lamda, **kwargs):
        super(CorrnetCost, self).__init__(**kwargs)
        self.lamda = lamda

    def cor(self,y1, y2, lamda):
        y1_mean = K.mean(y1, axis=0)
        y1_centered = y1 - y1_mean
        y2_mean = K.mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = K.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = K.sqrt(K.sum(y1_centered * y1_centered, axis=0) + 1e-8)
        corr_dr2 = K.sqrt(K.sum(y2_centered * y2_centered, axis=0) + 1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr / corr_dr
        return K.sum(corr) * lamda

    def call(self ,x ,mask=None):
        h1=x[0]
        h2=x[1]
        corr = self.cor(h1,h2,self.lamda)
        #self.add_loss(corr,x)
        #we output junk but be sure to use it for the loss to be added
        return corr
    def get_output_shape_for(self, input_shape):
        #print input_shape[0][0]
        return (input_shape[0][0],input_shape[0][1])

def corr_loss(y_true, y_pred):
    #print y_true.type,y_pred.type
    #return K.zeros_like(y_pred)
    return y_pred


class CorrNet:
    def __init__(self, data_l, data_r, Lambda=0.02, nb_epoch=10):
        self.data_l = data_l
        self.data_r = data_r
        self.Lambda = Lambda
        self.nb_epoch = nb_epoch
        
        dimx = self.data_l.shape[1]
        dimy = self.data_r.shape[1]
        
        inpx = Input(shape=(dimx,))
        inpy = Input(shape=(dimy,))
        
        hx = Dense(256, activation='relu')(inpx)
        hx = Dense(128, activation='relu')(hx)
        
        hy = Dense(256, activation='relu')(inpy)
        hy = Dense(128, activation='relu')(hy)
        
        h = Add()([hx,hy])
        
        recx = Dense(128, activation='relu')(h)
        recx = Dense(256, activation='relu')(recx)
        recx = Dense(dimx, activation='relu')(recx)
        
        recy = Dense(128, activation='relu')(h)
        recy = Dense(256, activation='relu')(recy)
        recy = Dense(dimy, activation='relu')(recy)    
        
        branchModel = Model([inpx,inpy], [recx,recy,h])
        
        [recx1,recy1,h1] = branchModel([inpx, ZeroPadding()(inpy)])
        [recx2,recy2,h2] = branchModel([ZeroPadding()(inpx), inpy])
        [recx3,recy3,h] = branchModel([inpx, inpy])
        corr = CorrnetCost(-Lambda)([h1,h2])
        
        opt = optimizers.Adam(lr=0.01)
        model = Model([inpx,inpy],[recx1,recx2,recx3,recy1,recy2,recy3,corr])
        model.compile(loss=["mse","mse","mse","mse","mse","mse",corr_loss], optimizer=opt)
        self.model = model
        self.branchModel = branchModel
        
    def train(self):
        data_l = self.data_l
        data_r = self.data_r
        nb_epoch = self.nb_epoch
        self.model.fit([data_l, data_r], 
                  [data_l,data_l,data_l,data_r,data_r,data_r,np.ones(data_l.shape)], epochs=nb_epoch)
    
    def left_to_right(self, new_data_l):
        branchModel = self.branchModel
        _,new_data_r,_ = branchModel.predict([new_data_l, np.zeros(new_data_l.shape)])
        return new_data_r
    
    def right_to_left(self, new_data_r):
        branchModel = self.branchModel
        new_data_l,_,_ = branchModel.predict([np.zeros(new_data_r.shape), new_data_r])
        return new_data_l
    
    def left_to_latent(self, new_data_l):
        branchModel = self.branchModel
        _,_,h = branchModel.predict([new_data_l, np.zeros(new_data_l.shape)])
        return h
        
    def right_to_latent(self, new_data_r):
        branchModel = self.branchModel
        _,_,h = branchModel.predict([np.zeros(new_data_r.shape), new_data_r])
        return h

    def both_to_latent(self, new_data_l, new_data_r):
        branchModel = self.branchModel
        _,_,h = branchModel.predict([new_data_l, new_data_r])
        return h        


class CorrTarget:
    def __init__(self, data_l, data_r, target, Lambda=0.02, nb_epoch=10):
        self.data_l = data_l
        self.data_r = data_r
        self.target = target
        self.nb_epoch = nb_epoch
        self.Lambda = Lambda
        
        dimx = self.data_l.shape[1]
        dimy = self.data_r.shape[1]        
        
        inpx = Input(shape=(dimx,))
        inpy = Input(shape=(dimy,))
        
        hx = Dense(256, activation='relu')(inpx)
        hx = Dense(128, activation='relu')(hx)
        
        hy = Dense(256, activation='relu')(inpy)
        hy = Dense(128, activation='relu')(hy)
        
        h = Add()([hx,hy])
        t = Dense(64, activation='relu')(h)
        t = Dense(32, activation='relu')(t)
        t = Dense(1, activation='relu')(t)
        
        branchModel = Model([inpx,inpy], [t,h])
        
        [t1,h1] = branchModel([inpx, ZeroPadding()(inpy)])
        [t2,h2] = branchModel([ZeroPadding()(inpx), inpy])
        [t3,h] = branchModel([inpx, inpy])
        corr = CorrnetCost(-Lambda)([h1,h2]) 
        
        opt = optimizers.Adam(lr=0.01)
        model = Model([inpx,inpy],[t1, t2, t3, corr])
        model.compile(loss=["mse","mse","mse",corr_loss], optimizer=opt)
        self.model = model
        self.branchModel = branchModel
        
    def train(self):
        data_l = self.data_l
        data_r = self.data_r
        target = self.target
        nb_epoch = self.nb_epoch
        self.model.fit([data_l, data_r], 
                  [target, target, target, np.ones(data_l.shape)], epochs=nb_epoch)
        
    def predict_by_left(self, new_data_l):
        branchModel = self.branchModel
        t,_ = branchModel.predict([new_data_l, np.zeros(new_data_l.shape)])
        return t
    
    def predict_by_right(self, new_data_r):
        branchModel = self.branchModel
        t,_ = branchModel.predict([np.zeros(new_data_r.shape), new_data_r])
        return t
    
    def right_to_latent(self, new_data_r):
        branchModel = self.branchModel
        _,h = branchModel.predict([np.zeros(new_data_r.shape), new_data_r])
        return h

    def both_to_latent(self, new_data_l, new_data_r):
        branchModel = self.branchModel
        _,h = branchModel.predict([new_data_l, new_data_r])
        return h       
    