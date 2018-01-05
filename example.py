import h5py
import lrsplines as lr
import numpy as np


# Load HDF5 file
h5 = h5py.File('filename.hdf5', 'r')

# Create LR Surface from data in HDF5 file
bytedata = np.array(h5['0/basis/Elasticity-1/1']).tobytes()
patch = lr.LRSurface.from_bytes(bytedata)

# Get parameter ranges
print(patch.start())
print(patch.end())

# Evalute at parameter values
print(patch(0.1, 0.2))

# Evaluate derivatives
print(patch(0.1, 0.2, d=(1,0)))
print(patch(0.1, 0.2, d=(0,1)))

# Set CPs to evaluate fields
cps = np.array(h5['0/1/displacement'])
patch.set_controlpoints(cps)
