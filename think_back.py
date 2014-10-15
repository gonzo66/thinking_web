# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:44:48 2014

@author: Gonzalo Vaca-Castano
Thinking with images . Processing in Python
"""

import os
from matplotlib import pyplot
import h5py
import pickle
import json
import time
import scipy.io as sio
import numpy as np

import skimage.io
import skimage.transform

from nearpy import Engine
import pandas as pd
from scipy import sparse
import tables, warnings
from scipy.sparse import lil_matrix
from collections import Counter


numwords=77488
contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/vocab.mat')
vocab=contents['vocab']

contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/kdist.mat');
kdist=contents['kdist']

contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/kcand.mat');
kcand=contents['kcand']


def eAnd(*args):
    return [all(tuple) for tuple in zip(*args)]


def eOr(*args):
    return [any(tuple) for tuple in zip(*args)]


def load_sparse_matrix(fname,mname) :
  warnings.simplefilter("ignore", UserWarning) 
  f = h5py.File(fname)
  #M = sparse.csc_matrix( (f.root.M.data[...], f.root.M.ir[...], f.root.M.jc[...]) )
  data = np.asarray(f[mname]["data"])
  ir = np.asarray(f[mname]["ir"])
  jc = np.asarray(f[mname]["jc"])    
  #M = sparse.coo_matrix(data, (ir, jc))  
  f.close()    
  #M = lil_matrix((len(jc)-1, max(ir)+1 ))
  #for iCount in range(len(jc)-1):
  #    startval=jc[iCount]
  #    endval=jc[iCount+1]
  #    M[iCount,ir[startval:endval]]=data[startval:endval]
  
  return data,ir,jc

def load_sparse_matrix2(fname,mname) :
  global numwords  
  warnings.simplefilter("ignore", UserWarning) 
  f = h5py.File(fname)
  #M = sparse.csc_matrix( (f.root.M.data[...], f.root.M.ir[...], f.root.M.jc[...]) )
  data = np.asarray(f[mname]["data"])
  ir = np.asarray(f[mname]["ir"])
  jc = np.asarray(f[mname]["jc"])     
  f.close()    
  #M = lil_matrix((len(jc)-1, max(ir)+1 ))
  imindex=np.zeros(np.shape(ir))
  for iCount in range(len(jc)-1):
      startval=jc[iCount]
      endval=jc[iCount+1]
      imindex[startval:endval]=iCount
  
  M = sparse.coo_matrix( (data, (ir, imindex)),shape=(numwords,1000000 ))
  M=M.tocsr()  
  return data,ir,jc,M


#dimension=4096

NN='data_171820,data_518482,data_891226,data_180327,data_865238,data_713745,data_745131,data_798342,data_790735,data_414952,data_796535,data_784321,data_108434,data_366884,data_706694,data_254362,data_562288,data_38281,data_862095,data_256675*data_497203,data_53940,data_197051,data_224213,data_747084,data_794752,data_235710,data_251909,data_939451,data_493155,data_122136,data_579685,data_377904,data_240873,data_339586,data_386017,data_600656,data_654593,data_405099,data_18384'
distances='69.0477,71.631,73.872,74.9348,75.2159,75.7679,76.0319,76.4538,78.1767,78.2648,78.8302,79.6723,82.0259,82.245,82.3034,82.4909,82.4923,82.5085,82.7844,83.0125*69.0617,69.5625,70.0957,70.5137,71.1337,73.7914,73.9076,74.3477,74.3525,74.8544,74.8926,74.9509,75.0385,75.3417,75.359,75.37,75.5802,75.7402,76.4037,76.6373'

NNtext = NN.split('*')
distancestext = distances.split('*')
NNs=[]
distances=[]

numquery=len(NNtext)

for iQuery in range(numquery):
    NNs.append([])
    distances.append([])
    queryNN=NNtext[iQuery]
    querydistances=distancestext[iQuery]
    NNdetails = queryNN.split(',')
    distancesdetails = querydistances.split(',')
    for iNN in range(len(NNdetails)):
        temp=NNdetails[iNN]
        temp2=temp.split('_')
        #toadd='/imgs/part%02d/' % (int(temp2[1])/100000)
        #toadd2= '%08d.jpg' %  (int(temp2[1])+1)
        #NNs[iQuery].append(toadd + toadd2)
        NNs[iQuery].append((int(temp2[1])+1))
        distances[iQuery].append(float(distancesdetails[iNN]))

distances = np.array(distances)

#print NNs[0][0] , NNs[0][1], NNs[1][1]
#print distances[0][0] , distances[0][1], distances[1][1]
#contents = h5py.File('/home/ashan/gonzalo/ECCV2014/BingChallenge/BoWT.mat')  
#BoWT=(contents['/BoWT/data'])

# Load the entire array into memory, like you're doing for matlab...
#A = sparse.csc_matrix((contents['/BoWT/data'], contents["BoWT"]["ir"], contents["BoWT"]["jc"]))
[data,ir,jc,M]=load_sparse_matrix2('/home/ashan/gonzalo/ECCV2014/BingChallenge/BoWT.mat','BoWT')

#load the words kernel (distances between words in the text space
contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/kdist.mat');
kdist=contents['kdist']

contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/kcand.mat');
kcand=contents['kcand']


contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/word_frequency.mat')
freq=contents['freq']

# Let's re weight the distances
nndists2=np.exp(-1*distances/np.mean(distances));
for i in range(numquery):
  nndists2[i]= np.arange(1, 0, -0.05)


# Step 1. Get a subset of candidate images to do retrieval that includes
# - Most representative words of candidate images
# - Words that are common between the closest words of a set of most similar images to queries

# Get textual representation (Weighting the words acording to the distance)
all_concepts=[]

#exclude= [1 ,2,3,4,5,6,9,15,16,41,	122,	259,	678,	995,	1201,	1594,	1712,	2058,	2064,	2139,	2427,	3084,	5726,	13393,	14467,	15683,	16860,	23151,	23222,	46378,	58587]
exclude=[0, 1, 2, 3, 4, 5, 8, 14, 15, 4, 121, 258, 677, 994, 1200, 1593, 1711, 2057, 2063, 2138, 2426, 3083, 5725, 13392, 14466, 15682, 16859, 23150, 23221, 46377, 58586]

candidates=[]
cand_representation=[]
mymax=0
for iConc in range(numquery):
    core_concepts=np.zeros((np.size(nndists2,1),numwords))
    startind=jc[np.array(NNs[iConc])]
    endind=jc[np.array(NNs[iConc])+1]
    # build core_concepts matrix. (text representation of the images closer to query image)
    for iNN in range(np.size(nndists2,1)):
        core_concepts[iNN,ir[startind[iNN]:endind[iNN]]]=data[startind[iNN]:endind[iNN]]
    
    core_concepts2= np.dot(nndists2[iConc,:] , core_concepts)    
    if (iConc==0) :   
        core_concepts3=core_concepts2
    else:
        core_concepts3=np.add(core_concepts3,core_concepts2)
    
    mymax= max(max(core_concepts2),mymax)
    cand_representation.append(core_concepts2)    
    core_concepts2[exclude]=0
    #I = argsort(a[:,i]), b=a[I,:]
    I = np.argsort(core_concepts2)
    I=I[-1::-1]    
    #cand=core_concepts2[I]   values of the most important concepts
    
    candidates.extend(I[0:10])    #Only 10 best words for each query candidate    
    all_concepts.extend(I[10:200].tolist())
    
for iCand in candidates:
  print(vocab[iCand][0][0])

knum=500
newcand=[]
for iC1 in range(4):    #Explore for the first 4 concepts
    distances1= kdist[1:knum,candidates[iC1]]
    candidates1=kcand[1:knum,candidates[iC1]]
    for iC2 in range(4):    #Explore for the first 4 concepts of second image
            distances2= kdist[1:knum,candidates[10+iC2]]
            candidates2=kcand[1:knum,candidates[10+iC2]]
            temp=np.intersect1d(candidates1, candidates2)
            if len(temp)>0:
                #erase elements from stop list
                to_erase=np.intersect1d(temp,exclude)
                to_erase=np.in1d(temp,exclude)                
                temp = temp[~to_erase]            
                newcand.extend(temp)
                candidates.extend(temp)
                
            
#if len(newcand)>0
                
            
for iCand in candidates:
    print(vocab[iCand][0][0])
                
    

#
listcounts=Counter(newcand)    # Histogram of the words used for the remaining 190 words
num_common=max(listcounts.values())   # the maximun value that a common word apperares in the query terms
#
## create empty list of new common words between retrieved images 
new_concept=np.zeros((1,numwords))
new_concept=new_concept[0]
#
## Find the word concept common between the two outputs
for k, v in listcounts.items(): 
  if v > 1: 
    if not (np.array(exclude)==k).any(): 
        new_concept[k]=v*(mymax+1)/num_common
        candidates.append(k)



# obtain a list of candidates to retrieve from.

#ic1=True
#for iCand in candidates:
#    if ic1 :
#        list_candidates = (ir == iCand)
#        ic1=False
#    else:
#        list_candidates=eOr(list_candidates , (ir == iCand))
posit=[]
for iCand in candidates:
    tempcand=np.where(ir==iCand)
    posit.extend(tempcand[0].tolist())
    
posit.sort()

list_candidates=[]
start_img=0
for itempcandidate in posit:
    while jc[start_img]<itempcandidate:
        start_img=start_img+1
    list_candidates.append(start_img)
list_candidates=np.unique(np.array(list_candidates))


# text representation of the current search 
text_representation=np.add(new_concept,core_concepts3)

# Perform tf-idf. Each non zero word determines a candidate set of images(from BoWT). 
# Get an score of the search representation on the candidate images 
scores=np.zeros((1,len(list_candidates)))
scores=scores[0]
for iConcept in text_representation.nonzero()[0]:
    tmpscore=(text_representation[iConcept]/freq[iConcept])* M[iConcept,list_candidates].toarray()[0]
    scores=np.add(scores,tmpscore)

# retrieve images
ind = np.argsort(scores)
ind=ind[-1::-1]
retrieved_imgs=list_candidates[ind[0:200]]
