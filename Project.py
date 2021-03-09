import cv2
import numpy as np
import multiprocessing as mp #Possible route for parallel programming
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing():
  img = cv2.imread('download (1).jpg', cv2.COLOR_BGR2RGB)
  
  #Blur and alter shape for ML model
  #To avoid overfitting and have proper format for training
  img = cv2.GaussianBlur(img, (3,3), 0)
  rows, cols, rgb = img.shape
  
  #cv2.imshow("Blurred Original", img)
  
  #Normalize RGB values to make composite for filter
  r = (img[:,:,2] - img[:,:,2].min())/(img[:,:,2].max() - img[:,:,2].min())
  g = (img[:,:,1] - img[:,:,1].min())/(img[:,:,1].max() - img[:,:,1].min())
  b = (img[:,:,0] - img[:,:,0].min())/(img[:,:,0].max() - img[:,:,0].min())
  
  #For training ML model
  reshaped_img = img.reshape(rows*cols, 3)
  
  #Merging channels to make composite image
  test = cv2.merge((r,g,b))
  #cv2.imshow("Test", test)
  
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
  
  binary_mask = binary_mask.reshape(rows, cols)
  #plt.title("Binary Mask of Original (Blurred)")
  #plt.imshow(binary_mask)
  #plt.show()
  
  test_binary_mask = test_binary_mask.reshape(rows, cols)
  #plt.title("Binary Mask of Test Image (Blurred)")
  #plt.imshow(test_binary_mask, cmap='tab20_r')
  #plt.show()
  
  #Convert numpy array to 8-bit for Canny Edge Detection
  binary_mask = np.uint8(binary_mask)
  test_binary_mask = np.uint8(binary_mask)

  return binary_mask, test_binary_mask

#Implements edge detection algorithm (Canny Algorithm)
def detectShoreline(image, test_image):
  #Get Edges and display
  edges = cv2.Canny(image, 2, 5)
  #cv2.imshow("Edges", edges)
  
  test_edges = cv2.Canny(test_image, 2, 5)
  #cv2.imshow("Test Edges", test_edges)
  
  #cv2.waitKey(0)
  
  #Save images to direcrtory
  cv2.imwrite("Original_Edges.png", edges)
  cv2.imwrite("Test_Edges.png", test_edges)
  

#Critical Section
#Parallelize functions with multiprocessing library (maybe) for dataset of images
startTime = time.time()
image, test_image = preprocessing()
detectShoreline(image, test_image)

print(time.time() - startTime)
