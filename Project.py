import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process
import threading
from sklearn.cluster import KMeans
import time
import os

os.chdir(r'C:\Users\kille\Desktop\UMBC\Junior Year\CMSC 483\CMSC 483 Project')

#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing(img, imageID):
  #Blur and alter shape for ML model
  #To avoid overfitting and have proper format for training
  img = cv2.GaussianBlur(img, (3,3), 0)
  rows, cols, rgb = img.shape
  
  #Normalize RGB values to make composite for filter
  r = (img[:,:,2] - img[:,:,2].min())/(img[:,:,2].max() - img[:,:,2].min())
  g = (img[:,:,1] - img[:,:,1].min())/(img[:,:,1].max() - img[:,:,1].min())
  b = (img[:,:,0] - img[:,:,0].min())/(img[:,:,0].max() - img[:,:,0].min()) * 1.7
  
  #For training ML model
  reshaped_img = img.reshape(rows*cols, 3)
  
  #Merging channels to make composite image
  test = cv2.merge((r,g,b))
  
  #For training model on Composite
  test_reshaped = test.reshape(rows*cols, 3)
  
  #Making binary np array of coastline
  kmeans = KMeans(2)
  kmeans.fit(reshaped_img)
  binary_mask = kmeans.predict(reshaped_img)
  
  #Binary mask of composite
  kmeans = KMeans(2)
  kmeans.fit(test_reshaped)
  test_binary_mask = kmeans.predict(test_reshaped)
  
  #Get binary masks
  binary_mask = binary_mask.reshape(rows, cols)
  test_binary_mask = test_binary_mask.reshape(rows, cols)
  
  #Convert numpy array to 8-bit for Canny Edge Detection
  binary_mask = np.uint8(binary_mask)
  test_binary_mask = np.uint8(binary_mask)

  detectShoreline(binary_mask, test_binary_mask, imageID)

#Implements edge detection algorithm (Canny Algorithm)
def detectShoreline(image, test_image, imageID):
  #Get Edges and display
  edges = cv2.Canny(image, 2, 5)
  test_edges = cv2.Canny(test_image, 2, 5)  
  cv2.waitKey(0)
  
  #Save images to direcrtory
  cv2.imwrite("Edge images/"+str(imageID)+'.jpg', edges)
  cv2.imwrite('Test Edge images/Test_'+str(imageID)+'.jpg', test_edges)

#Load in all images in directory
images = [] #List for all images to be read
count = 0
for imagePath in os.listdir('images'):
  images.append(cv2.imread('images/'+imagePath, cv2.COLOR_BGR2RGB))

#Begin preprocessing and shoreline detection
startTime = time.time()
for i in range(len(images)):
  preprocessing(images[i], i)
print('Serial Execution Time:', time.time() - startTime)

#Begin Parallel Execution
iterable = [(images[i], i) for i in range(len(images))]
#Make threads here
startTime = time.time()
threads = [threading.Thread(target=preprocessing, args=(i[0], i[1])) for i in iterable]
for t in threads:
  t.start()
for t in threads:
  t.join()
print('Parallel Execution Time:', time.time() - startTime)

#Works with code snippet below to parallelize serial version
def parallelize(images_subset):
  for i in range(len(images_subset)):
    preprocessing(images_subset[i], i)

#Other Parallel method
num_threads = 4
interval = int(len(iterable)/num_threads)
startTime = time.time()
threads = [threading.Thread(target=parallelize, args=(images[i*interval:(i + 1)*interval],)) for i in range(num_threads)]

#Error Case
if num_threads > len(images):
  num_threads = len(images)

for t in threads:
  t.start()
for t in threads:
  t.join()
print('Alt Parallel Algo Time: ', time.time() - startTime)