import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop

fh=Dataset("modConvectiveDbase.nc")
zKu=fh["zKu"][:]
piaKu=fh["piaKu"][:]
zKa=fh["zKa"][:]
sfcPrecip=fh["sfcPrecip"][:]
fh.close()
#stop
    
from sklearn.preprocessing import StandardScaler
scalerZKu = StandardScaler()
scalerZKa = StandardScaler()
scalerPrec=StandardScaler()


scalerZKu.fit(zKu[:,:])
scalerZKa.fit(zKa[:,:])
#scalerPrec.fit(pRate[:,:])
zKu_sc=scalerZKu.transform(zKu)[:,:]
zKa_sc=scalerZKa.transform(zKa)[:,:]
#pRate_sc=scalerPrec.transform(pRate)[:,np.newaxis]
pRate_sc=np.array([sfcPrecip/piaKu,piaKu]).T
from sklearn.model_selection import train_test_split

ind=range(zKu.shape[0])

nt,nz=zKu_sc.shape
ind_train, ind_test,\
    y_train, y_test = train_test_split(range(nt), pRate_sc,\
                                       test_size=0.5, random_state=42)


from sklearn.cluster import KMeans
import matplotlib


import scipy
import scipy.optimize
from scipy.optimize import minimize as minimize

from sklearn.neighbors import NearestNeighbors
import numpy as np
X=zKu_sc
x_train=X[ind_train,:]
x_test=X[ind_test,:]


def lstm_model(ndims=2,nout=2):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(nout, recurrent_activation=None, \
                               return_sequences=False)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


itrain=0
#stop
model=lstm_model(1,2)

if itrain==1:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(x_train[:,-60:], y_train[:], \
                        batch_size=32,epochs=40,
                        validation_data=(x_test[:,-60:], \
                                         y_test[:]))
else:
    model=tf.keras.models.load_model("radarProfilingKuFreq_modConv.h5")
