import h5py
filename = 'train_data_processed.h5'
import numpy as np
h5f = h5py.File(filename, 'r')
# get a List of data sets in group 'dd48'
a_dset_keys = list(h5f['table'].keys())

# Get the data
for dset in a_dset_keys :
    ds_data = (h5f['table'][dset])
    print ('dataset=', dset)
    print (ds_data.dtype)
    if ds_data.dtype == 'float64' :
        csvfmt = '%.18e'
    elif ds_data.dtype == 'int64' :
        csvfmt = '%.10d'
    else:
        csvfmt = '%s'
    np.savetxt('output_'+dset+'.csv', ds_data, fmt=csvfmt, delimiter=',')

from pandas import HDFStore
store = HDFStore(filename)
store['table'].to_csv('output.csv')