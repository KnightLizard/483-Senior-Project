import cv2
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
import threading
from sklearn.cluster import KMeans
import time
import multiprocessing
import os

os.chdir(r'C:\Users\kille\Desktop\UMBC\Junior Year\CMSC 483\CMSC 483 Project')
total_size = 100

#Load in all images in directory
images = [] #List for all images to be read
for imagePath in os.listdir('sat'):
  images.append(cv2.imread('sat/'+imagePath, cv2.IMREAD_GRAYSCALE))

#Implements edge detection algorithm (Canny Algorithm)
def detectShoreline(image, id, left, right):
  #Get Edges and display
  edges = cv2.Canny(image, 2, 5)
  #print(left, right)  
  
  #Save images to direcrtory
  #cv2.imwrite("Edge images/"+str(imageID)+'.jpg', edges)
  images[id][left:right] = edges

#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing(id, left, right, doKMeans = False):
    #Blur to reduce noise
    img = images[id][left:right]
    test = cv2.GaussianBlur(img, (13,13), 0)
    
    #Water in the data set is mostly gray value 28, 45 reduces noise
    ret, test = cv2.threshold(test, 28, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    test_edges = cv2.Canny(test, 2, 5)

    #print(left, right)

    detectShoreline(test_edges, id, left, right)

#Begin Serial preprocessing and shoreline detection
def serial():
  startTime = time.time()
  endTime = 0
  for i in range(len(images)):
    preprocessing(i, 0, len(images[i]))
    endTime = time.time()
    cv2.imwrite('sat serial{}.jpg'.format(i),images[i])
  
  print('Serial Execution Time:', endTime - startTime, 'Mb/s:', total_size/(endTime - startTime))

#Works with code snippet below to parallelize serial version
def parallelize(id, left, right):
    preprocessing(id, left, right)

#Parallel method
def alternateParallel(num_threads = 1):
  total_time = 0
  #Partition Data
  for i in range(len(images)):
    threads = []
    row, col = images[i].shape
    #print(row, col)
    interval = int(row/num_threads)
    for j in range(num_threads):
        if ((i + 1) * interval) >= row:
            threads.append(threading.Thread(target=parallelize, args=(i,j*interval,col)))
        #break
        else:
            threads.append(threading.Thread(target=parallelize, args=(i,j*interval,(j + 1)*interval)))
    startTime = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time += time.time() - startTime
    cv2.imwrite('sat{} num_threads {}.jpg'.format(i, num_threads), images[i])

  print('Num Threads:', num_threads, 'Alt Parallel Algo Time: ', total_time, 'Mb/s:', total_size/(total_time))

serial()
alternateParallel(16)