import cv2
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
import threading
from sklearn.cluster import KMeans
import time
import multiprocessing
import os

absPath = os.path.abspath(__file__)
absPath = absPath[0:absPath.rindex('\\')]
os.chdir(absPath)
#os.chdir(r'C:\Users\kille\Desktop\UMBC\Junior Year\CMSC 483\CMSC 483 Project')
total_size = 151

#Load in all images in directory
images = [] #List for all images to be read

def loadImages():
    for imagePath in os.listdir('sat'):
        images.append(cv2.imread('sat/'+imagePath, cv2.IMREAD_GRAYSCALE))

#Implements edge detection algorithm (Canny Algorithm)
def detectShoreline(image, id, left, right):
  #Get Edges and display
  edges = cv2.Canny(image, 2, 5)
  #Save images to direcrtory
  images[id][left:right] = edges

#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing(id, left, right):
    #Blur to reduce noise
    img = images[id][left:right]
    test = cv2.GaussianBlur(img, (13,13), 0)
    
    #Water in the data set is mostly gray value 28, 45 reduces noise
    ret, test = cv2.threshold(test, 32, 255, cv2.THRESH_BINARY)
    #ret, test = cv2.threshold(test, 45, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #test_edges = cv2.Canny(test, 2, 5)

    detectShoreline(test, id, left, right)

#Begin Serial preprocessing and shoreline detection
def serial():
  startTime = time.time()
  endTime = 0
  for i in range(len(images)):
    preprocessing(i, 0, len(images[i]))
    endTime = time.time()
    #cv2.imwrite('satOut/sat{}serial.jpg'.format(i),images[i])
  for i in range(len(images)):
    cv2.imwrite('satOut/sat{}serial.jpg'.format(i),images[i])
  
  print('Serial Execution Time:', endTime - startTime, 'Mb/s:', total_size/(endTime - startTime))

#Works with code snippet below to parallelize serial version
def parallelize(id, left, right):
    #preprocessingPar(id, left, right, imagesIn)
    preprocessing(id, left, right)
    #time.sleep(1)

def worker(rank):
    imagesAlt = []
    count = 0
    for imagePath in os.listdir('sat'):
        if(count % var_dict['numProcs'] == rank):
            imagesAlt.append(cv2.imread('sat/'+imagePath, cv2.IMREAD_GRAYSCALE))
        count = count + 1
            
    for i in range(len(imagesAlt)):
        img = imagesAlt[i]
        test = cv2.GaussianBlur(img, (13,13), 0)
    
        #Water in the data set is mostly gray value 28, 45 reduces noise
        ret, test = cv2.threshold(test, 32, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(test, 2, 5)
        imagesAlt[i] = edges
    for i in range(len(imagesAlt)):
        cv2.imwrite('satOut/sat{}serial.jpg'.format(i * var_dict['numProcs'] + rank),imagesAlt[i])

#set up shared data between the processes
var_dict = {}
def initWorkerData(numProcs):
    var_dict['numProcs'] = numProcs
    
    
#File decomposition parallel method
def fileLevelParallel(num_threads = 1):
  startTime = time.time()
  
  numProcesses = num_threads
  with multiprocessing.Pool(processes=numProcesses, initializer=initWorkerData, initargs=(numProcesses,)) as pool:
      pool.map(worker, range(numProcesses))
      
  endTime = time.time()
  total_time = endTime - startTime
  
  print('Num Threads:', num_threads, 'File Parallel Algo Time: ', total_time)#, 'Mb/s:', total_size/(total_time2))
    
#Data decomposition parallel method
def alternateParallel(num_threads = 1):
  total_time = 0
  
  #Partition Data
  startTime = time.time()
  for i in range(len(images)):
    threads = []
    row, col = images[i].shape
    #print(row, col)
    interval = int(row/num_threads)
    for j in range(num_threads):
        if (j == num_threads - 1):
            threads.append(threading.Thread(target=parallelize, args=(i,j*interval,row)))
        else:
            threads.append(threading.Thread(target=parallelize, args=(i,j*interval,(j + 1)*interval)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
  endTime = time.time()
  
  total_time = endTime - startTime
  for i in range(len(images)):
    cv2.imwrite('satOut/sat{}num_threads{}.jpg'.format(i, num_threads), images[i])

  print('Num Threads:', num_threads, 'Alt Parallel Algo Time: ', total_time, 'Mb/s:', total_size/(total_time))


if (__name__ == '__main__'):
    startTotSerTime = time.time()
    loadImages()
    serial()
    endTotSerTime = time.time()
    print('Serial total Time:', endTotSerTime - startTotSerTime)
    
    images = []
    startTotFileTime = time.time()
    fileLevelParallel(1)
    endTotFileTime = time.time()

    images = []
    startTotParTime = time.time()
    loadImages()
    alternateParallel(1)
    endTotParTime = time.time()
    
    print("File Parallel total Time (cores:", 1 , "):", endTotFileTime - startTotFileTime, \
          "\nData Parallel total Time(cores:", 1 , "):", endTotParTime - startTotParTime, "\n")
            
    for i in range(2, 10, 2):
        images = []
        startTotFileTime = time.time()
        fileLevelParallel(i)
        endTotFileTime = time.time()
    
        images = []
        startTotParTime = time.time()
        loadImages()
        alternateParallel(i)
        endTotParTime = time.time()
        
        print("File Parallel total Time (cores:", i , "):", endTotFileTime - startTotFileTime, \
          "\nData Parallel total Time(cores:", i , "):", endTotParTime - startTotParTime, "\n")
    
   