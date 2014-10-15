from bottle import route, run, template
from bottle import static_file
from bottle import route, request, response
from bottle import redirect
from bottle import get, post, request
import os
import caffe
from matplotlib import pyplot
import h5py
import pickle
import json
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import EuclideanDistance
from nearpy.storage import MemoryStorage
from nearpy.storage import GonzaloStorage
import time
import scipy.io as sio
import numpy as np

import skimage.io
import skimage.transform

from nearpy import Engine
import pandas as pd
from scipy import sparse
import warnings
from collections import Counter


dimension=4096
numwords=77488

#define engine and storage
fp = open('engine.txt', 'r')
engine = pickle.load(fp)
fp.close()

# CNN definitions
NET = None
IMAGE_DIM = 256
CROPPED_DIM = 227
IMAGE_CENTER = None
IMAGE_MEAN = None
CROPPED_IMAGE_MEAN = None
BATCH_SIZE = None
NUM_OUTPUT = None
CROP_MODES = ['list', 'center_only', 'corners', 'selective_search']

contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/word_frequency.mat')
freq=contents['freq']

contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/vocab.mat')
vocab=contents['vocab']

exclude=[0, 1, 2, 3, 4, 5, 8, 14, 15, 4, 121, 258, 677, 994, 1200, 1593, 1711, 2057, 2063, 2138, 2426, 3083, 5725, 13392, 14466, 15682, 16859, 23150, 23221, 46377, 58586]

# Think definitions
#load textual representation of the input vocabulary %BoWT is an array of  78K x 1M
#a=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/BoWT.mat')	
#BoWT=(a['BoWT'])



def load_sparse_matrix2(fname,mname) :
  """
  Load sparse matrix saved in matlab v7.3
  Input:
    fname: string with path to mat file
    mname: name of the sparse matrix
  Output:
    data,ir,jc,M: M is the sparse matrix 
  """
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

# Load the entire array into memory, like you're doing for matlab...
[data,ir,jc,M]=load_sparse_matrix2('/home/ashan/gonzalo/ECCV2014/BingChallenge/BoWT.mat','BoWT')

#load the words kernel (distances between words in the text space
contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/kdist.mat');
kdist=contents['kdist']

contents=sio.loadmat('/home/ashan/gonzalo/ECCV2014/BingChallenge/kcand.mat');
kcand=contents['kcand']




def load_image(filename):
  """
  Input:
    filename: string

  Output:
    image: an image of size (H x W x 3) of type uint8.
  """
  img = skimage.io.imread(filename)
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img


def format_image(image, window=None, cropped_size=False):
  """
  Input:
    image: (H x W x 3) ndarray
    window: (4) ndarray
      (ymin, xmin, ymax, xmax) coordinates, 0-indexed
    cropped_size: bool
      Whether to output cropped size image or full size image.

  Output:
    image: (3 x H x W) ndarray
      Resized to either IMAGE_DIM or CROPPED_DIM.
    dims: (H, W) of the original image
  """
  dims = image.shape[:2]

  # Crop a subimage if window is provided.
  if window is not None:
    image = image[window[0]:window[2], window[1]:window[3]]

  # Resize to input size, subtract mean, convert to BGR
  image = image[:, :, ::-1]
  if cropped_size:
    image = skimage.transform.resize(image, (CROPPED_DIM, CROPPED_DIM)) * 255
    image -= CROPPED_IMAGE_MEAN
  else:
    image = skimage.transform.resize(image, (IMAGE_DIM, IMAGE_DIM)) * 255
    image -= IMAGE_MEAN

  image = image.swapaxes(1, 2).swapaxes(0, 1)
  return image, dims


def _image_coordinates(dims, window):
  """
  Calculate the original image coordinates of a
  window in the canonical (IMAGE_DIM x IMAGE_DIM) coordinates

  Input:
    dims: (H, W) of the original image
    window: (ymin, xmin, ymax, xmax) in the (IMAGE_DIM x IMAGE_DIM) frame

  Output:
    image_window: (ymin, xmin, ymax, xmax) in the original image frame
  """
  h, w = dims
  h_scale, w_scale = h / IMAGE_DIM, w / IMAGE_DIM
  if h_scale !=0  and w_scale !=0:
      image_window = window * np.array((1. / h_scale, 1. / w_scale,
                                   h_scale, w_scale))
  else:
      image_window = np.array((1, 1, h, w))
                                       
  return image_window.round().astype(int)


def config(model_def, pretrained_model, gpu, image_dim, image_mean_file):
  global IMAGE_DIM, CROPPED_DIM, IMAGE_CENTER, IMAGE_MEAN, CROPPED_IMAGE_MEAN
  global NET, BATCH_SIZE, NUM_OUTPUT

  # Initialize network by loading model definition and weights.
  t = time.time()
  print("Loading Caffe model.")
  NET = caffe.CaffeNet(model_def, pretrained_model)
  NET.set_phase_test()
  NET.set_mode_cpu()
  if gpu:
    NET.set_mode_gpu()
  print("Caffe model loaded in {:.3f} s".format(time.time() - t))

  # Configure for input/output data
  IMAGE_DIM = image_dim
  CROPPED_DIM = NET.blobs[0].width
 # CROPPED_DIM = 227
  IMAGE_CENTER = int((IMAGE_DIM - CROPPED_DIM) / 2)

    # Load the data set mean file
  IMAGE_MEAN = np.load(image_mean_file)

  CROPPED_IMAGE_MEAN = IMAGE_MEAN[IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  :]
                                  
 
  NUM_OUTPUT=NET.blobs[-1].channels
  BATCH_SIZE = NET.blobs[0].num  # network batch size
  #NUM_OUTPUT = NET.blobs()[-1].channels  # number of output classes




def _assemble_images_list(input_df):
  """
  For each image, collect the crops for the given windows.

  Input:
    input_df: pandas.DataFrame
      with 'filename', 'ymin', 'xmin', 'ymax', 'xmax' columns

  Output:
    images_df: pandas.DataFrame
      with 'image', 'window', 'filename' columns
  """
  # unpack sequence of (image filename, windows)
  windows = input_df[['ymin', 'xmin', 'ymax', 'xmax']].values
  image_windows = (
    (ix, windows[input_df.index.get_loc(ix)]) for ix in input_df.index.unique()
  )

  # extract windows
  data = []
  for image_fname, windows in image_windows:
    image = load_image(image_fname)
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_center_only(image_fnames):
  """
  For each image, square the image and crop its center.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  crop_start, crop_end = IMAGE_CENTER, IMAGE_CENTER + CROPPED_DIM
  crop_window = np.array((crop_start, crop_start, crop_end, crop_end))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    data.append({
      'image': image[np.newaxis, :,
                     crop_start:crop_end,
                     crop_start:crop_end],
      'window': _image_coordinates(dims, crop_window),
      'filename': image_fname
    })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_corners(image_fnames):
  """
  For each image, square the image and crop its center, four corners,
  and mirrored version of the above.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  # make crops
  indices = [0, IMAGE_DIM - CROPPED_DIM]
  crops = np.empty((5, 4), dtype=int)
  curr = 0
  for i in indices:
    for j in indices:
      crops[curr] = (i, j, i + CROPPED_DIM, j + CROPPED_DIM)
      curr += 1
  crops[4] = (IMAGE_CENTER, IMAGE_CENTER,
              IMAGE_CENTER + CROPPED_DIM, IMAGE_CENTER + CROPPED_DIM)
  all_crops = np.tile(crops, (2, 1))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    image_crops = np.empty((10, 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    curr = 0
    for crop in crops:
      image_crops[curr] = image[:, crop[0]:crop[2], crop[1]:crop[3]]
      curr += 1
    image_crops[5:] = image_crops[:5, :, :, ::-1]  # flip for mirrors
    for i in range(len(all_crops)):
      data.append({
        'image': image_crops[i][np.newaxis, :],
        'window': _image_coordinates(dims, all_crops[i]),
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_selective_search(image_fnames):
  """
  Run Selective Search window proposals on all images, then for each
  image-window pair, extract a square crop.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  windows_list = selective_search.get_windows(image_fnames)

  data = []
  for image_fname, windows in zip(image_fnames, windows_list):
    image = load_image(image_fname)
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def compute_feats(images_df):
  input_blobs = [np.ascontiguousarray(
    np.concatenate(images_df['image'].values), dtype='float32')]
  output_blobs = [np.empty((BATCH_SIZE, NUM_OUTPUT, 1, 1), dtype=np.float32)]

  NET.Forward(input_blobs, output_blobs)
  
  #NET.Forward_Till_Layer(input_blobs, output_blobs,'drop7')
  
  # 4096 layer
  feats= NET.blobs[-3].data

  # final layer  
  #feats= NET.blobs[-1].data

  feats=feats.squeeze()
  feats = [feats[i].flatten() for i in range(len(output_blobs[0]))]

  # Add the features and delete the images.
  del images_df['image']
  images_df['feat'] = feats
  return images_df



def assemble_batches(inputs, crop_mode='center_only'):
  """
  Assemble DataFrame of image crops for feature computation.

  Input:
    inputs: list of filenames (center_only, corners, and selective_search mode)
      OR input DataFrame (list mode)
    mode: string
      'list': take the image windows from the input as-is
      'center_only': take the CROPPED_DIM middle of the image windows
      'corners': take CROPPED_DIM-sized boxes at 4 corners and center of
        the image windows, as well as their flipped versions: a total of 10.
      'selective_search': run Selective Search region proposal on the
        image windows, and take each enclosing subwindow.

  Output:
    df_batches: list of DataFrames, each one of BATCH_SIZE rows.
      Each row has 'image', 'filename', and 'window' info.
      Column 'image' contains (X x 3 x CROPPED_DIM x CROPPED_IM) ndarrays.
      Column 'filename' contains source filenames.
      Column 'window' contains [ymin, xmin, ymax, xmax] ndarrays.
      If 'filename' is None, then the row is just for padding.

  Note: for increased efficiency, increase the batch size (to the limit of gpu
  memory) to avoid the communication cost
  """
  if crop_mode == 'list':
    images_df = _assemble_images_list(inputs)

  elif crop_mode == 'center_only':
    images_df = _assemble_images_center_only(inputs)

  elif crop_mode == 'corners':
    images_df = _assemble_images_corners(inputs)

  elif crop_mode == 'selective_search':
    images_df = _assemble_images_selective_search(inputs)

  else:
    raise Exception("Unknown mode: not in {}".format(CROP_MODES))

  # Make sure the DataFrame has a multiple of BATCH_SIZE rows:
  # just fill the extra rows with NaN filenames and all-zero images.
  N = images_df.shape[0]
  remainder = N % BATCH_SIZE
  if remainder > 0:
    zero_image = np.zeros_like(images_df['image'].iloc[0])
    zero_window = np.zeros((1, 4), dtype=int)
    remainder_df = pd.DataFrame([{
      'filename': None,
      'image': zero_image,
      'window': zero_window
    }] * (BATCH_SIZE - remainder))
    images_df = images_df.append(remainder_df)
    N = images_df.shape[0]

  # Split into batches of BATCH_SIZE.
  ind = np.arange(N) / BATCH_SIZE
  df_batches = [images_df[ind == i] for i in range(N / BATCH_SIZE)]
  return df_batches









# Method to display the images
@route('/static/<filename:path>')
def server_static(filename):
    return static_file(filename, root='/home/ashan/gonzalo/bottle/')

# Clear the cookies for the current user
@route('/clear', method='POST')
def clear():
   # response.set_cookie('counter', '0')
   # response.set_cookie('querys', '')
   # num_queries=len(querystext)
    return show_example()


#clear the cookies for the current user
@route('/clear')
def clear2():
   # response.set_cookie('counter', '0')
   # response.set_cookie('querys', '')
   # num_queries=len(querystext)
    return show_example()



#Upload page to the web server
@route('/upload', method='POST')
def do_upload():
    upload     = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'

    save_path = 'queries/' + upload.filename
    upload.save(save_path, overwrite=True) # appends upload.filename automatically
    
    #save counter and list of uploaded images
    count = int( request.cookies.get('counter', '0') )
    count += 1
    query = request.cookies.get('querys', '')
    if query == '':
	querys= upload.filename
    else:
	querys= query + ',' + upload.filename

    #print querys
		
    response.set_cookie('counter', str(count))
    response.set_cookie('querys', querys)
    querystext = querys.split(',')
    #num_queries=len(querystext)

    # obtain feature using CNN
    # save_path='/queries/img9.jpg'
    MODEL_FILE = '/home/ashan/gonzalo/bottle/imagenet/imagenet_deploy.prototxt'
    PRETRAINED = '/home/ashan/gonzalo/bottle/imagenet/caffe_reference_imagenet_model'
    IMAGE_FILE = '/home/ashan/gonzalo/bottle/' + save_path
    MEAN_FILE= '/home/ashan/gonzalo/caffe/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy'

    #config(args.model_def, args.pretrained_model, args.gpu, args.images_dim, args.images_mean_file)
    config(MODEL_FILE, PRETRAINED, False, 256, MEAN_FILE)
    images=[]
    images.append(IMAGE_FILE)
    image_batches = assemble_batches(images, 'center_only')


    dfs_with_feats = []
    dfs_with_feats.append(compute_feats(image_batches[0]))
    df = pd.concat(dfs_with_feats).dropna(subset=['filename'])
    df.set_index('filename', inplace=True)
  

    # Label coordinates
    coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
    df[coord_cols] = pd.DataFrame(
    data=np.vstack(df['window']), index=df.index, columns=coord_cols)
    del(df['window'])

    prediction=df['feat'].values.tolist()
    prediction=prediction[0]

#    prediction=myfeats['feat'].values.tolist()
#    a=sio.loadmat('/home/ashan/gonzalo/bottle/out.mat')
#    prediction=(a['feat_Query'][0])


    # calculate Nearest neighbors output (LSH) of the query descriptor
    # print len(prediction)
    N = engine.neighbours(prediction)
    #print len(N)
    #print len(N[0])		


    # convert Nearest neighbors output (LSH) to text for web processing
    currentNN=''
    currentdist=''
    for n in range(len(N)):
       temp=N[n]	
       currentNN=currentNN +  temp[1] + ',' 
       currentdist= currentdist + str(temp[2]) + ','
    
    currentNN=currentNN[0:(len(currentNN)-1)]
    currentdist=currentdist[0:(len(currentdist)-1)]

	
    # get NN and distances from cookies
    NN = request.cookies.get('NN', '')	
    distances = request.cookies.get('distances', '')	

    if NN == '':
	print 'NN is empty'
        NN= currentNN
        distances=currentdist
    else:
        NN= NN + '*' + currentNN
        distances= distances + '*' + currentdist 
		
    response.set_cookie('NN', NN)
    response.set_cookie('distances', distances)
    print NN
    print distances   

 

    NNtext = NN.split('*')
    distancestext = distances.split('*')
    #NNs=numpy.zeros(shape=(num_queries,20))
    #distances=numpy.zeros(shape=(num_queries,20))
    NNs=[]
    NNs2=[]    
    distances=[]
     # Added to see the labels of the found images
    candidates=[]
    cand_representation=[]
    
    for iQuery in range(len(NNtext)):
        NNs.append([])
        NNs2.append([])
        distances.append([])       	
        queryNN=NNtext[iQuery]
        querydistances=distancestext[iQuery]
        NNdetails=queryNN.split(',')
        distancesdetails=querydistances.split(',')
        for iNN in range(len(NNdetails)):
            temp=NNdetails[iNN]
            temp2=temp.split('_')
            toadd='/imgs/part%02d/' % (int(temp2[1])/100000)
            toadd2= '%08d.jpg' %  (int(temp2[1])+1)
            NNs[iQuery].append(toadd + toadd2)
            NNs2[iQuery].append((int(temp2[1])))
            distances[iQuery].append(float(distancesdetails[iNN]))  

        core_concepts= np.zeros((len(NNs2[iQuery]),numwords))
        startind=jc[np.array(NNs2[iQuery])]
        endind=jc[np.array(NNs2[iQuery])+1]
        # build core_concepts matrix. (text representation of the images closer to query image)
        for iNN in range(len(NNs2[iQuery])):
            core_concepts[iNN,ir[startind[iNN]:endind[iNN]]]=data[startind[iNN]:endind[iNN]]
            core_concepts[iNN,exclude]=0
            I = np.argsort(core_concepts[iNN,:])
            I=I[-1::-1]
            #candidates.extend(I[0:10])    #Only 10 best words for each query candidate    
            #for iCand in candidates:
            print(" ")
            for iCand in I[0:10]:    
                print(vocab[iCand][0][0])
        
    
#        cand_representation.append(core_concepts)          
#        I = np.argsort(core_concepts)
#        I=I[-1::-1]    
        

    output = template('front_end2', outlabel='Press process to see the full outputs', count=count , querys=querystext, NNs=NNs , distances=distances)
    return output



#Do conceptual search
@route('/search', method='POST')
def do_search():
    count = int( request.cookies.get('counter', '0') )
    querys= request.cookies.get('querys', '') 
    querystext = querys.split(',')
    num_queries=len(querystext)
 # get NN and distances from cookies
    NN = request.cookies.get('NN', '')	
    distances = request.cookies.get('distances', '')


    NNtext = NN.split('*')
    distancestext = distances.split('*')
    NNs=[]    
    NNs2=[]    
    distances=[]
    
    for iQuery in range(len(NNtext)):
        NNs.append([])
        NNs2.append([])
        distances.append([])       	
        queryNN=NNtext[iQuery]
        querydistances=distancestext[iQuery]
        NNdetails = queryNN.split(',')
        distancesdetails = querydistances.split(',')
        for iNN in range(len(NNdetails)):
           temp=NNdetails[iNN]
           temp2=temp.split('_')
           toadd='/imgs/part%02d/' % (int(temp2[1])/100000)
           toadd2= '%08d.jpg' %  (int(temp2[1])+1)
           NNs2[iQuery].append(toadd + toadd2)
           NNs[iQuery].append((int(temp2[1])))
           distances[iQuery].append(float(distancesdetails[iNN]))

    numquery=len(NNtext)
    distances = np.array(distances)
    
    
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
    
    #the real one
    #exclude=[0, 1, 2, 3, 4, 5, 8, 14, 15, 4, 121, 258, 677, 994, 1200, 1593, 1711, 2057, 2063, 2138, 2426, 3083, 5725, 13392, 14466, 15682, 16859, 23150, 23221, 46377, 58586]
    
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
            core_concepts[iNN,exclude]=0
        
        core_concepts2= np.dot(nndists2[iConc,:] , core_concepts)    
        if (iConc==0) :   
            core_concepts3=core_concepts2
        else:
            core_concepts3=np.add(core_concepts3,core_concepts2)
    
        mymax= max(max(core_concepts2),mymax)
        cand_representation.append(core_concepts2)    
        core_concepts2[exclude]=0
        core_concepts3[exclude]=0
        #I = argsort(a[:,i]), b=a[I,:]
        I = np.argsort(core_concepts2)
        I=I[-1::-1]    
        #cand=core_concepts2[I]   values of the most important concepts
        numcandidates=min(len(core_concepts2.nonzero()[0]),10)
    
        candidates.extend(I[0:numcandidates])    #Only 10 best words for each query candidate    
#        all_concepts.extend(I[10:200].tolist())
  
    #candidates contain the first 10 better tag descriptors of each query  
 #   for iCand in candidates:
 #      print(vocab[iCand][0][0])
        
        
    # Find new candidates using the current tag descriptors !! 
    knum=200
    newcand=[]
    for iC1 in range(4):    #Explore for the first 4 concepts
        #distances1= kdist[1:knum,candidates[iC1]]
        candidates1=kcand[1:knum,candidates[iC1]]
        for iC2 in range(4):    #Explore for the first 4 concepts of second image
            #distances2= kdist[1:knum,candidates[10+iC2]]
            candidates2=kcand[1:knum,candidates[10+iC2]]
            temp=np.intersect1d(candidates1, candidates2)
            if len(temp)>0:
                #erase elements from stop list
                to_erase=np.intersect1d(temp,exclude)
                to_erase=np.in1d(temp,exclude)                
                temp = temp[~to_erase]            
                newcand.extend(temp)
                candidates.extend(temp)
    
    
    for iCand in candidates:
        print(vocab[iCand][0][0])

    print(" ")
    
    for iCand in newcand:
        print(vocab[iCand][0][0])


#    # create empty list of new common words between retrieved images 
    new_concept=np.zeros((1,numwords))
    new_concept=new_concept[0]

    if len(newcand)>0:
        listcounts=Counter(newcand)    # Histogram of the words used for the remaining 190 words
        num_common=max(listcounts.values())   # the maximun value that a common word apperares in the query terms
    #    
       
        # Find the word concept common between the two outputs
        for k, v in listcounts.items(): 
          #if v > 1: 
            if not (np.array(exclude)==k).any(): 
                new_concept[k]=v*(mymax+1)/num_common
                #candidates.append(k)
    
    
    
    # obtain a list of candidates to retrieve from.    

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
    
    print(" ")
    I = np.argsort(core_concepts3)
    I=I[-1::-1]
    numcandidates=min(len(core_concepts3.nonzero()[0]),10)

    for iCand in I[0:numcandidates]:    
        print(vocab[iCand][0][0])

    print(" ")
    
    I = np.argsort(new_concept)
    I=I[-1::-1]
    numcandidates=min(len(new_concept.nonzero()[0]),10)
    for iCand in I[0:numcandidates]:    
        print(vocab[iCand][0][0])  
        
    print(" ")
    I = np.argsort(text_representation)
    I=I[-1::-1]
    numcandidates=min(len(text_representation.nonzero()[0]),30)
    for iCand in I[0:numcandidates]:    
        print (vocab[iCand][0][0])  
    
        
    
    # Perform tf-idf. Each non zero word determines a candidate set of images(from BoWT). 
    # Get an score of the search representation on the candidate images 
    scores=np.zeros((1,len(list_candidates)))
    scores=scores[0]
#    for iConcept in text_representation.nonzero()[0]:    
#        tmpscore=(text_representation[iConcept]/freq[iConcept])* M[iConcept,list_candidates].toarray()[0]
    for iConcept in core_concepts3.nonzero()[0]:    
        tmpscore=(core_concepts3[iConcept]/freq[iConcept])* M[iConcept,list_candidates].toarray()[0]        
        scores=np.add(scores,tmpscore)
    
    # retrieve images
    ind = np.argsort(scores)
    ind=ind[-1::-1]
    retrieved_imgs=list_candidates[ind[0:200]]
    
    think_out=[]
    for iImg in retrieved_imgs:
		toadd='/imgs/part%02d/' % (int(iImg)/100000)
		toadd2= '%08d.jpg' %  (int(iImg)+1)	
		think_out.append(toadd + toadd2)
		
    
    

    output = template('front_end3', outlabel='Press process to see the full outputs', count=count , querys=querystext, NNs=NNs2 , think_img=think_out)
    return output
#    return 'Do search here'





# The front End
@route('/example')
def show_example():

    response.set_cookie('NN', '')
    response.set_cookie('distances', '')
    response.set_cookie('querys', '')
    response.set_cookie('counter', '0')

    output = template('front_end')
    return output

run(host='0.0.0.0', port=8080)
