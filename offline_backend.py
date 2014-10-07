import numpy
from matplotlib import pyplot
import h5py
import pickle

import json
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import EuclideanDistance
from nearpy.storage import MemoryStorage
from nearpy import Engine

#from redis import Redis
#from nearpy.storage import RedisStorage
from nearpy.storage import GonzaloStorage

#load the visual features of all the images from the dataset
featIN=h5py.File('featIN.mat')['featIN']

#Create binary projections and save them in HD
rbp = RandomBinaryProjections('rbp', 10)
dimension=4096

#Trying redis
gonzalo_storage = GonzaloStorage()
engine = Engine(dimension, lshashes=[rbp], distance=EuclideanDistance(),vector_filters=[NearestFilter(20)],  storage=gonzalo_storage)

fp = open('engine.txt', 'w')
pickle.dump(engine, fp)
fp.close()

#engine = Engine(dimension, lshashes=[rbp])
for index in range(1000000):
 v=featIN[range(dimension),index]
 #v=numpy.float16(featIN[range(dimension),index])
 engine.store_vector(v, 'data_%d' % index)

engine.save_all()
