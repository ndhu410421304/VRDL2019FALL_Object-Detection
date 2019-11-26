import scipy.io
import pandas as pd

import hdf5storage
mat = hdf5storage.loadmat('digitStruct.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.iteritems()})
data.to_csv("example.csv")