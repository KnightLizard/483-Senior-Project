import cv2
import numpy as np
import multiprocessing as mp #Possible route for parallel programming
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import multiprocessing

#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing():
    #some images to try, the threshold is only designed for the satellite data (not the thumbnail)
    #img = cv2.imread('download (2).jpg', cv2.IMREAD_GRAYSCALE) 
    img = cv2.imread('images/LC08/LC08_L2SP_022029_20130329_20200912_02_T1_SR_B5.jpg', cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread('LC08_L2SP_022029_20130329_20200912_02_T1_SR_B5_quart.jpg', cv2.IMREAD_GRAYSCALE) #quarter size
    #img = cv2.imread('LC08_L2SP_022029_20130329_20200912_02_T1_thumb_large.jpeg', cv2.IMREAD_GRAYSCALE) #colored 
  
    return img

#Implements edge detection algorithm (Canny Algorithm)
#  Either uses thresholding or clustering prior to Canny, choose with doKMeans
def detectShoreline(img, doKMeans = False):  
    if(doKMeans):
        #Blur to reduce noise
        img = cv2.GaussianBlur(img, (13,13), 0)
        rows, cols = img.shape
  
        #For training ML model
        reshaped_img = img.reshape(rows*cols, 1)
  
        #Making binary np array of coastline
        kmeans = KMeans(2)
        kmeans.fit(reshaped_img)
        binary_mask = kmeans.predict(reshaped_img)
  
        
        binary_mask = binary_mask.reshape(rows, cols)
  
  
        #Convert numpy array to 8-bit for Canny Edge Detection
        binary_mask = np.uint8(binary_mask)
        
        #Get Edges and display
        edges = cv2.Canny(binary_mask, 2, 5)
        return edges
    else:
        #Blur to reduce noise
        test = cv2.GaussianBlur(img, (13,13), 0)
        #Water in the data set is mostly gray value 28, 45 reduces noise
        ret, test = cv2.threshold(test,45,255,cv2.THRESH_BINARY)
      
        test_edges = cv2.Canny(test, 2, 5)
        return test_edges
  
#set up shared data between the processes
var_dict = {}
def initWorkerData(data, img_shape, numProcs):
    var_dict['data'] = data
    var_dict['img_shape'] = img_shape
    var_dict['numProcs'] = numProcs

#Data decomposistion based parallel application of the alg. Applies it to p
#  horizontal sections of the image.
def worker(rank):
    #name = multiprocessing.current_process().name
    #print(name, 'Starting')
    dataArr = np.frombuffer(var_dict['data']).reshape(var_dict['img_shape'])
    return detectShoreline((dataArr[(int)((rank / var_dict['numProcs']) * var_dict['img_shape'][0]) : \
                (int)(((rank + 1) / var_dict['numProcs']) * var_dict['img_shape'][0]), :]).astype(np.uint8), False)

#Critical Section
#Parallelize functions with multiprocessing library () for dataset of images
if (__name__ == '__main__'):
    startTime = time.time()
    image = preprocessing()
    print(image.shape)
    
    img_shape = image.shape
    data = multiprocessing.RawArray('d', img_shape[0] * img_shape[1])
    dataArr = np.frombuffer(data).reshape(img_shape)
    np.copyto(dataArr, image)
    
    numProcesses = 4
    with multiprocessing.Pool(processes=numProcesses, initializer=initWorkerData, initargs=(data, img_shape, numProcesses)) as pool:
        result = pool.map(worker, range(numProcesses))
    
    dataArr = (np.concatenate(result).reshape(img_shape))
    cv2.imwrite("Edge images/Original_Edges.png", dataArr)
    print(time.time() - startTime)
