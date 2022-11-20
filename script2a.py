#!/usr/bin/env python
from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import process_time    #checking cpu time
from sklearn.preprocessing import MinMaxScaler 
import time
from rnn_models_cpu import *
#load the data
data =pd.read_csv("tndata.csv")
# tt, II, RR =data_process(data,scaler, 'no')
scaler =MinMaxScaler()

#define the preprocess function
def data_process(data, scaler, cs ="yes"):
    #reverse index
    tdata=data.reindex(index=data.index[::-1])
    #get the infected 
    I=tdata['TOTAL_CONFIRMED']
    #get the recovered
    R =tdata['TOTAL_INACTIVE_RECOVERED']
    #get the length of the data
    nn =len(I)
    # show whether want to scaling
    if cs =="yes": ##indicate yes
        tt=np.linspace(0,nn, nn)
        y1 =np.array(I[:nn]).reshape((-1,1))
        y2 =np.array(R[:nn]).reshape((-1,1))
        #scaling
        II =scaler.fit_transform(y1)
        RR =scaler.fit_transform(y2)
        #plot
        plt.plot(tt, II, '--r')
        plt.plot(tt, RR, '--b')
        plt.legend(['Infected', 'Recovered'])
        plt.title('Scaled Data') 
        plt.show()
    else:  ##indicate no
        tt=np.linspace(0,nn, nn)
        II =np.array(I[:nn]).reshape((-1,1))
        RR =np.array(R[:nn]).reshape((-1,1))
        #plot
        plt.plot(tt, II, '--r')
        plt.plot(tt, RR, '--b')
        plt.legend(['Infected', 'Recovered'])
        plt.title('UnScaled Data') 
        plt.show()    
    return tt, II, RR
p=2

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
name = MPI.Get_processor_name()
if rank == 0:
    tt, II, RR =data_process(data,scaler, 'no')
    i_true_l, i_pred_l, _, _, _, _,ep, i_loss_l=run_model_cpu(II, 'LSTM', 1500, 128, 500, 0.8) 
    i_true_b, i_pred_b, _, _, _, _,_, i_loss_b=run_model_cpu(II, 'BiLSTM', 1500, 128, 500, 0.8)
    i_true_g, i_pred_g, _, _, _, _,_, i_loss_g=run_model_cpu(II, 'GRU', 1500, 100, 500, 0.8)
    x_t =np.arange(0, len(i_pred_l),1).reshape((-1,1))
    
    
    plt.plot(x_t, i_true_l, '-o')
    plt.plot(x_t, i_pred_l, '--b')
    plt.plot(x_t, i_pred_b, '--r')
    plt.plot(x_t, i_pred_g, '--c')
    plt.legend(['LSTM', 'BiLSTM', 'GRU'])
    plt.title('Modeling with MPI on CPU')
    plt.savefig('wmpi_cpu_{}.png'.format(p))
    plt.show()
    print("rank0 finished")
