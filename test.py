# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 17:30:34 2018
@author: Admin
"""
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from DeNN import denoise_model,denoise_loss,denoise_model_general,readMatVars
datapath = 'U:/ADNI2_analysis/Denoise/ssegdata'
from os import listdir
from os.path import isfile, join
datafiles = [f for f in listdir(datapath) if isfile(join(datapath,f))]
import numpy
from scipy.io import savemat
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os.path as path
from scipy.stats.mstats import zscore
epochs = 50
success_subject = numpy.zeros((len(datafiles),1))
for subid,sub in enumerate(datafiles):
    tempdatadir = datapath+"/"+sub   
    savedatadir = datapath+"_DeNNTS/v1b1/"+sub[:-4]
    if path.isfile(savedatadir+'.mat')==True:
        savedatadir = datapath+"_DeNNTS/v1b1/"+sub[:-4]
        fMRIdata,c1T1,c2T1_erode,c3T1_erode = readMatVars(tempdatadir,varname=('fMRIdata','c1T1','c2T1_erode',
                                                                                       'c3T1_erode'))
        
        fMRIdata = numpy.transpose(fMRIdata,axes=(3,0,1,2))
        mask = numpy.mean(fMRIdata,axis=0)>0.05*numpy.max(fMRIdata)
        fMRIdata_q = fMRIdata[:,mask>0].T
        fMRIdata_q = numpy.reshape(zscore(fMRIdata_q,axis=1),fMRIdata_q.shape+(1,))
        
        c1T1_float = c1T1>0.9
        c23T1 = (c2T1_erode+c3T1_erode)>0.1
        fMRIdata_c1 = fMRIdata[:,numpy.logical_and(mask,c1T1_float)>0].T
        fMRIdata_c23 = fMRIdata[:,numpy.logical_and(mask,c23T1)>0].T
    
        fMRIdata_c1 = numpy.reshape(zscore(fMRIdata_c1,axis=1),fMRIdata_c1.shape+(1,))
        fMRIdata_c23 = numpy.reshape(zscore(fMRIdata_c23,axis=1),fMRIdata_c23.shape+(1,))
        nvoxel_c23 = fMRIdata_c23.shape[0]
        nvoxel_c1 = fMRIdata_c1.shape[0]
        nvoxel_train = numpy.min([nvoxel_c23,nvoxel_c1])
        
        trainind_c1 = numpy.random.permutation(nvoxel_c1)[:nvoxel_train]
        trainind_c23 = numpy.random.permutation(nvoxel_c23)[:nvoxel_train]
        tdim = fMRIdata_c1.shape[1]
        
        model = denoise_model_general(tdim,layers_type=["tden","tdis","tdis","conv","conv","conv"],layers_size=[128,32,16,8,4,1])
        opt = Adam(lr=0.05,beta_1=0.9, beta_2 = 0.999, decay = 0.05)
        model.compile(optimizer=opt,loss=denoise_loss)
        
        train_c1 = fMRIdata_c1[trainind_c1,:,:]
        train_c23= fMRIdata_c23[trainind_c23,:,:]
        y_true = numpy.ones((nvoxel_train,tdim,2))#fMRIdata_c23_strict[trainind_c23,:,:]#
        history = model.fit([train_c1[:,[i],:] for i in range(tdim)]+
                            [train_c23[:,[i],:] for i in range(tdim)],
                            y=y_true,batch_size = 500,validation_split=0.1,epochs = epochs)  
        fMRIdata_q_output = model.predict([fMRIdata_q[:,[i],:] for i in range(tdim)]+
                                            [fMRIdata_q[:,[i],:] for i in range(tdim)]
                                            ,batch_size=500)
        loss = history.history['loss']
        valloss = history.history['val_loss']
        savemat(savedatadir+'.mat',{'fMRIdata_q':fMRIdata_q_output[:,:,0],
                                    'mask':mask,
                                    'loss':loss,'valloss':valloss})
