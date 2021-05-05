#File: Large image Final method.py
#Description: A serial and two parallel implementations of the shoreline
#  detection algorithm designed for use with the Landsat data set.

import cv2
import multiprocessing
import threading
import time
import os

#change working directory to the root directory of the program
absPath = os.path.abspath(__file__)
absPath = absPath[0:absPath.rindex('\\')]
os.chdir(absPath)
total_size = 151 #directory size for bandwidth calculations

#Load in all images in directory
images = [] #List for all images to be read

#Populates the images array with all images in the sat directory
def loadImages():
    for imagePath in os.listdir('sat'):
        images.append(cv2.imread('sat/'+imagePath, cv2.IMREAD_GRAYSCALE))


#Implements edge detection algorithm (Canny Algorithm)
def detectShoreline(image, id, left, right):
  #Get Edges and display
  edges = cv2.Canny(image, 2, 5)
  #Save result back to images array
  images[id][left:right] = edges


#Preprocesses image for land-water classification and KMeans clustering
#Returns a binary mask of the image
def preprocessing(id, left, right):
    #Blur to reduce noise
    img = images[id][left:right]
    test = cv2.GaussianBlur(img, (13,13), 0)
    
    #Water in the data set is mostly gray value 28, 45 reduces noise
    ret, test = cv2.threshold(test, 32, 255, cv2.THRESH_BINARY)

    #Apply canny algorithm
    detectShoreline(test, id, left, right)


#Begin Serial preprocessing and shoreline detection
def serial():
    #Time the calculation across all images
    startTime = time.time()
    endTime = 0
    for i in range(len(images)):
        preprocessing(i, 0, len(images[i]))
        endTime = time.time()
        
    #write images to disk (not included in calculation time)
    for i in range(len(images)):
        cv2.imwrite('satOut/sat{}serial.jpg'.format(i),images[i])
  
    print('Serial Execution Time:', endTime - startTime, 'Mb/s:', total_size/(endTime - startTime))


#Works with code snippet below to parallelize serial version
#Used with data decomposition method
def parallelize(id, left, right):
    preprocessing(id, left, right)


#Process of the parallel loading, calculation, and saving version
def worker(argsList):
    rank = argsList[0]
    numProcs = argsList[1]
    imagesAlt = [] 
    
    #load a fraction of the images in the directory (distributed amongst the processes)
    count = 0
    for imagePath in os.listdir('sat'):
        if(count % numProcs == rank):
            imagesAlt.append(cv2.imread('sat/'+imagePath, cv2.IMREAD_GRAYSCALE))
        count = count + 1
            
    #apply the shoreline detection algorithm
    for i in range(len(imagesAlt)):
        img = imagesAlt[i]
        test = cv2.GaussianBlur(img, (13,13), 0)
    
        #Water in the data set is mostly gray value 28, 45 reduces noise
        ret, test = cv2.threshold(test, 32, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(test, 2, 5)
        imagesAlt[i] = edges
        
    #save the subset of images back to disk
    for i in range(len(imagesAlt)):
        cv2.imwrite('satOut/sat{}serial.jpg'.format(i * numProcs + rank),imagesAlt[i])

    
#File decomposition parallel method
def fileLevelParallel(num_threads = 1):
  startTime = time.time()
  
  #Set up a multiprocessing pool
  numProcesses = num_threads
  with multiprocessing.Pool(processes=numProcesses) as pool:
      pool.map(worker, zip(range(numProcesses), [numProcesses] * numProcesses))
      
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
      
    #each thread works on a separate series of rows in the image
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
    #save the images back to disk
    for i in range(len(images)):
        cv2.imwrite('satOut/sat{}num_threads{}.jpg'.format(i, num_threads), images[i])

    print('Num Threads:', num_threads, 'Alt Parallel Algo Time: ', total_time, 'Mb/s:', total_size/(total_time))

#Run and time the total time to load, calculate, and save for each version
if (__name__ == '__main__'):
    startTotSerTime = time.time()
    loadImages()
    serial()
    endTotSerTime = time.time()
    print('Serial total Time:', endTotSerTime - startTotSerTime)
    
    #File decomposition version does loading itself, so don't preload the images
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
    
   