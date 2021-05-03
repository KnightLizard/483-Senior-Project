import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
from sklearn.cluster import KMeans
import time
import os

#Change string to absolute directory on your machine
os.chdir(r'C:\Users\kille\Desktop\UMBC\Junior Year\CMSC 483\CMSC 483 Project')
total_size = 1.67 #MB of data in image directory

#Implements edge detection algorithm (Canny Algorithm)
def detectShoreline(image, imageID):
  #Get Edges and display
  edges = cv2.Canny(image, 2, 5)  
  cv2.waitKey(0)
  
  #Save images to direcrtory
  cv2.imwrite("Edge images/"+str(imageID)+'.jpg', edges)

#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing(img, imageID, doKMeans = False):
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
   
    detectShoreline(edges, imageID)
    
  else:
    #Blur to reduce noise
    test = cv2.GaussianBlur(img, (13,13), 0)
    
    #Water in the data set is mostly gray value 28, 45 reduces noise
    ret, test = cv2.threshold(test, 45, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    test_edges = cv2.Canny(test, 2, 5)

    detectShoreline(test_edges, imageID)


#Load in all images in directory
images = [] #List for all images to be read
images_ID = []
count = 0

for imagePath in os.listdir('images'):
  images.append(cv2.imread('images/'+imagePath, cv2.IMREAD_GRAYSCALE))
  images_ID.append(count)
  count += 1

#Begin Serial preprocessing and shoreline detection
def serial():
  startTime = time.time()
  for i in range(len(images)):
    preprocessing(images[i], i)
  endTime = time.time()
  print('Serial Execution Time:', endTime - startTime, 'Mb/s:', total_size/(endTime - startTime))

#Begin Parallel Execution
iterable = [(images[i], i) for i in range(len(images))]

#Works with code snippet below to parallelize serial version
def parallelize(images_subset, image_ID):
  for i in range(len(images_subset)):
    preprocessing(images_subset[i], image_ID[i])

#Parallel method
def alternateParallel(num_threads = 1):
  #num_threads = 5 #Change number of threads in parallel execution
  interval = int(len(iterable)/num_threads) #Size of grouping for each processor
  threads = []

  #Error Case
  if num_threads > len(images):
    num_threads = len(images)
    interval = 1

  #Partition Data
  for i in range(num_threads):
    if ((i + 1) * interval) >= len(images):
      threads.append(threading.Thread(target=parallelize, args=(images[i*interval:], images_ID[i*interval:])))
      #break
    else:
      threads.append(threading.Thread(target=parallelize, args=(images[i*interval:(i + 1)*interval], images_ID[i*interval:(i + 1)*interval])))

  startTime = time.time()
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  endTime = time.time()
  print('Num Threads:', num_threads, 'Alt Parallel Algo Time: ', endTime - startTime, 'Mb/s:', total_size/(endTime - startTime))

serial()
#originalParallel()
alternateParallel()
for i in range(2, 34, 2):
  alternateParallel(i)