#!/usr/bin/env python
from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import process_time    #checking cpu time
from sklearn.preprocessing import MinMaxScaler 
import time

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
name = MPI.Get_processor_name()
if rank == 0:
    i_true_l, i_pred_l, _, _, _, _,ep, i_loss_l=run_model(II, 'LSTM', 1500, 128, 500, 0.8) 
    i_true_b, i_pred_b, _, _, _, _,_, i_loss_b=run_model(II, 'BiLSTM', 1500, 128, 500, 0.8)
    i_true_g, i_pred_g, _, _, _, _,_, i_loss_g=run_model(II, 'GRU', 1500, 100, 500, 0.8)
    x_t =np.arange(0, len(i_pred_l),1).reshape((-1,1))
    print(ep)
    
    plt.plot(x_t, i_true_l, '-o')
    plt.plot(x_t, i_pred_l, '--b')
    plt.plot(x_t, i_pred_b, '--r')
    plt.plot(x_t, i_pred_g, '--c')
    plt.legend(['LSTM', 'BiLSTM', 'GRU'])
    plt.title('Infected Data')
    plt.show()
    print("rank0 finished")
