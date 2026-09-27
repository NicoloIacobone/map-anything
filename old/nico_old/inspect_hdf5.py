import h5py

path = '/scratch2/nico/distillation/dataset/DL3DV.hdf5'

with h5py.File(path, 'r') as f:
    f.visititems(lambda name, obj: print(name) if len(name.split('/')) < 3 else None)