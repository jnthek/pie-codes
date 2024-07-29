#! /usr/bin/env python3

import numpy as np
import argparse
import h5py
import sys
sys.path.append('../')
from scint_tools import scint_utils

parser = argparse.ArgumentParser(description='Combine two hdf5 files into one')
parser.add_argument('-f1', metavar='filename1', help='First HDF5 filename')
parser.add_argument('-f2', metavar='filename2', help='Second HDF5 filename, should be newer than f1')
parser.add_argument('-o', metavar='filenameOut', help='Output HDF5 filename')

args = parser.parse_args()
fname_a = args.f1
fname_b = args.f2
fname_out = args.o

mydata_a = scint_utils.scintdata(fname_a)
mydata_b = scint_utils.scintdata(fname_b)

if ((mydata_a.Nchan != mydata_b.Nchan ) 
            or not (np.isclose(mydata_a.f_low, mydata_b.f_low)) or not (np.isclose(mydata_a.f_high, mydata_b.f_high))
            or not (np.isclose(mydata_a.delta_f, mydata_b.delta_f)) 
            or not (np.isclose(mydata_a.f_high, mydata_b.f_high))
            or not (np.isclose(mydata_a.t_aver, mydata_b.t_aver))
            or (mydata_b.timestamps[0] < mydata_a.timestamps[-1])):
    print ("Files are not compatible for merging, exiting ...")
    exit(1)

else:
    print ("Merging")
    print ("Writing to",fname_out)
    hf = h5py.File(fname_out, 'w')

    data_group = hf.create_group('data')
    data_group.attrs["NFFT"] = mydata_a.Nchan
    data_group.attrs["Ntimes"] = mydata_a.Ntimes + mydata_b.Ntimes
    data_group.attrs["f_low"] = mydata_a.f_low
    data_group.attrs["f_high"] = mydata_a.f_high
    data_group.attrs["delta_f"] = mydata_a.delta_f
    data_group.attrs["t_start"] = mydata_a.tstart_unix_time
    data_group.attrs["t_end"]   = mydata_b.tend_unix_time #Should be the second one ..
    data_group.attrs["t_aver"]  = mydata_a.t_aver
    data_group.attrs["t_scan"]  = (mydata_a.t_scan+mydata_a.t_scan)/2

    timestamps = np.zeros(mydata_a.Ntimes + mydata_b.Ntimes)
    timestamps[0:mydata_a.Ntimes] = mydata_a.timestamps
    timestamps[mydata_a.Ntimes:] = mydata_b.timestamps
    data_group.create_dataset('timestamps', data=timestamps)

    data_group.create_dataset('radio', data=mydata_a.radio_data/1e6, maxshape=(None,None)) #Division by a million is required

    data_group['radio'].resize((mydata_a.Ntimes + mydata_b.Ntimes), axis=0)
    data_group['radio'][-mydata_b.Ntimes:] = mydata_b.radio_data/1e6
    # data_group.create_dataset('radio', data=mmaped_file) #, chunks=mmaped_file.shape, compression='gzip', compression_opts=9
    hf.close()
    print ("Combined files into one")
