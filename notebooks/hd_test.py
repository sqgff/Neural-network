from rbm import RBM
from autoencoder_rbm import Autoencoder_RBM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('../data/64times_overlap.csv')
df.drop("timestamp", inplace=True, axis=1)
#df.dropna(inplace=True)
min_max_scaler = MinMaxScaler()
df = min_max_scaler.fit_transform(df)

autoencoder = Autoencoder_RBM(rbm_layers=[1000, 500, 100],
                              rbm_gauss_visible=True,
                              finetune_num_epochs=50,
                              do_pretrain=False,
                              finetune_loss_func='mse')


autoencoder.fit(np.array(df))
