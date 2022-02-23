from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import numpy as np
from hb_sub import *
import glob

fs=glob.glob("dsrt/newSubset/ku*dsrt*")
count=0
count_s=0
pia1=[]
pia2=[]
zKuL=[]
zKaL=[]
cfadZ=np.zeros((50,47),float)
cfadZKa=np.zeros((50,47),float)
sfcPrecipL=[]
srtPIAL=[]
for f in sorted(fs):
    fh=Dataset(f)
    sfcPrecip=fh["sfcPrecip"][:]
    drelFact=fh["drelFact"][:]
    relFact=fh["relFact"][:]
    dsrtPIA=fh["dsrtPIA"][:]
    srtPIA=fh["srtPIA"][:]
    bzd=fh["bzd"][:]
    bcf=fh["bcf"][:]
    zKu=fh["zKu"][:]
    zKa=fh["zKa"][:]
    for i,relF in enumerate(drelFact):
        if bzd[i]+30>bcf[i] or bzd[i]>160 or bcf[i]<168:
            continue
        if sfcPrecip[i]<20 and ((relF==1) or (relF==2)):
            count+=1
            zKuL.append(zKu[i,bzd[i]+16-80:bzd[i]+30:2])
            zKaL.append(zKa[i,bzd[i]+16-80:bzd[i]+30:2])
            sfcPrecipL.append(sfcPrecip[i])
            zKu1= zKuL[-1]
            zKa1= zKaL[-1]
            srtPIAL.append(dsrtPIA[i,0])
            for k in range(47):
                if zKu1[k]>10 and zKa1[k]>10:
                    if k>1 and k<46:
                        if zKu1[k-1]<10 or zKu1[k+1]<10:
                            continue
                    iz=int(zKuL[-1][k])
                    if iz<50:
                        cfadZ[iz,k]+=1
                if zKa1[k]>12 and zKu1[k]>10:
                    iz=int(zKaL[-1][k])
                    if iz<50:
                        cfadZKa[iz,k]+=1
            if relFact[i]==1 or relFact[i]==2:
                count_s+=1
                pia1.append(dsrtPIA[i,0])                
                pia2.append(srtPIA[i])

plt.subplot(121)
plt.pcolormesh(range(50),range(47)[::-1],cfadZ.T,cmap='jet',norm=cols.LogNorm())
plt.xlim(10,45)
plt.subplot(122)
plt.pcolormesh(range(50),range(47)[::-1],cfadZKa.T,cmap='jet',norm=cols.LogNorm())
plt.xlim(10,40)

plt.figure()
from minisom import MiniSom
n1=49
n2=1
nz=zKu1.shape[0]
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)



zKuL=np.array(zKuL)
zKuL[zKuL<0]=0
zKaL=np.array(zKaL)
zKaL[zKaL<0]=0
som.random_weights_init(zKuL)
som.train_random(zKuL,500)
nt=zKuL.shape[0]
winL=np.zeros((nt),int)
it=0
for z1 in zKuL:
    win=som.winner(z1)
    winL[it]=win[0]
    it+=1

for i in range(n1):
    ind1=np.nonzero(winL==i)
    #plt.plot(np.array(zSimL[i]).mean(axis=0),range(47))
    zm=np.array(zKuL[ind1[0],:]).mean(axis=0)
    plt.plot(zm,range(47)[::-1])
    #print(len(ind1[0]))


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scalerZKu = StandardScaler()
scalerPrec=StandardScaler()

zKu=zKuL
sfcPrecip=np.array(sfcPrecipL)
scalerZKu.fit(zKuL[:,:])


from sklearn.model_selection import train_test_split
nt=sfcPrecip.shape[0]
ind_train, ind_test,\
    y_train, y_test = train_test_split(range(nt), sfcPrecip,\
                                       test_size=0.5, random_state=42)


ind=range(zKu.shape[0])

srtPIAL=np.array(srtPIAL)

from sklearn.cluster import KMeans
import matplotlib
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
n_neighbors=45
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
knn.fit(zKu[ind_train,:], sfcPrecip[ind_train])
#.predict(T)
yp=knn.predict(zKu[ind_test,:])
y_test=sfcPrecip[ind_test]

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=50, \
                        algorithm='ball_tree').fit(zKu[ind_train,:][:,:])

rms,ind=nbrs.kneighbors(np.array(zKu[ind_test,:]))


yL=[]
sfcPrecip_mt=sfcPrecip[ind_train]
y_test=sfcPrecip[ind_test]

from sklearn.ensemble import RandomForestRegressor
est = RandomForestRegressor(n_estimators=20)
#zKaL[:,0]=srtPIAL
#est.fit(zKaL[ind_train,:], sfcPrecip[ind_train])

#yL=est.predict(zKaL[ind_test,:])
#stop
#yp=yL

import xarray as xr
zKuDB=xr.DataArray(zKuL)
zKaDB=xr.DataArray(zKaL)
sfcPrecipDB=xr.DataArray(sfcPrecipL)
piaDB=xr.DataArray(srtPIAL)
d=xr.Dataset({"zKu":zKuDB,"zKa":zKaDB,\
              "piaKu":piaDB, "sfcPrecip":sfcPrecipDB})
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in d.data_vars}
d.to_netcdf("modConvectiveDbase.nc", encoding=encoding)
