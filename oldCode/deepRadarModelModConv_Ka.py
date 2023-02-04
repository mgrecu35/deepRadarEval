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
X=zKa_sc
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


itrain=1
#stop
model=lstm_model(1,2)
itrain=0
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
    model=tf.keras.models.load_model("radarProfilingKaFreq_modConv.h5")

model.save("radarProfilingKaFreq_modConv.h5")
import pickle
yp=model(x_test[:,-60:])
pickle.dump({"y_pred":yp.numpy(),"y_test":y_test,"piaKu":piaKu},open("ka_output.pklz","wb"))

    

yp=yp.numpy()[:,0]*yp.numpy()[:,1]
y_testp=y_test[:,0]*piaKu[ind_test]
for intv in [[1,2],[2,4],[4,6],[6,8],[8,10],[10,20]]:
    a=np.nonzero((y_testp-intv[0])*(y_testp-intv[1])<0)
    #print(len(a[0]))
    rms=(((yp[a[0]]-y_testp[a[0]])**2).mean())**0.5/y_testp[a[0]].mean()
    print("%6.2f %6.2f %6.2f %6.2f"%(intv[0],intv[1],\
                                     (-1+yp[a[0]].mean()/y_testp[a[0]].mean())*100,rms*100))

yp_2=model(x_test[:,-40:])
yp=yp_2.numpy()[:,0]*piaKu[ind_test]
piaP=yp_2.numpy()[:,1]
y_testp=y_test[:,0]*piaKu[ind_test]
for intv in [[1,2],[2,4],[4,6],[6,8],[8,10],[10,20]]:
    a=np.nonzero((y_testp-intv[0])*(y_testp-intv[1])<0)
    #print(len(a[0]))
    rms=(((yp[a[0]]-y_testp[a[0]])**2).mean())**0.5/y_testp[a[0]].mean()
    print("%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f"%(intv[0],intv[1],\
                                                 (-1+yp[a[0]].mean()/y_testp[a[0]].mean())*100,rms*100,\
                                                 piaP[a[0]].mean(),piaKu[ind_test][a].mean()))

stop
sfcPrecip_0=scalerPrec.scale_[0]*yp[:,-1,0].numpy()+scalerPrec.mean_[0]
sfcPrecip_test=pRate[:,0][ind_test]
interVs=[[1,10],[10,20],[20,30],[30,40],[40,50]]
for int1 in interVs:
    a=np.nonzero((sfcPrecip_test-int1[0])*(sfcPrecip_test-int1[1])<0)
    diff=np.abs((sfcPrecip_test[a[0]]-sfcPrecip_0[a[0]]))
    #plt.hist(diff/sfcPrecip_test[a[0]]*100)
    b=np.nonzero(diff>1)
    print(int1, len(b[0])/len(a[0]))
