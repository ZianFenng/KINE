# Course Project of Hybrid Computing for Signal and Data Processing
# EECS E4750, Fall 2016, Columbia University 
# Project Name: Kinect Color and Depth Image Alignment
# Project Code: KINE
# Memebers: Tianqi Fang(tf2393)
#			Xiangfeng Gong(xg2244)

##################################################
## Code Starts Here
##################################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For use of matrix
import numpy as np

# The module below is used to mark time stamp
import time
	
# Import plotlibrary for plotting timing curve
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

# Import PYCUDA modules and libraries
from pycuda import driver, compiler, gpuarray, tools
# # -- initialize the device
import pycuda.autoinit

# Import signal to use the 2D conv function
from scipy import signal

# For image processing
from PIL import Image

#kernel
kernel_code ="""
// Define the constants will be used in the kernel

// Mask size, used for bilateral filter, and the boundary = maskSize / 2
#define boundary 4
#define maskSize 9

// The size of the depth image to test this demo code
#define picH 300
#define picW 430

// The size of the extended image used for bilateral filter 
#define eImgH picH*maskSize
#define eImgW picW*maskSize

// Define the mathematical constant PI
#define PI 3.1415926f

// Define directions for searching the nearest nonblack pixel for hole filling
#define direction 4 



// The directions of the four numbers stand for.
// Definition  0:left, 1:up, 2:right, 3:down
// Each thread deal with one direction for each pixel of the original image
__global__ void getDepthFill(const float* __restrict__ originalImg, float* depthFill)
{
	// This function is searching the nearest non-black pixel for every black pixel
	// And use the value of the non-black pixel to fill this pixel


	// Get the location of the pixel in the depth image and well as the direction of searching
	int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int globalIdy = threadIdx.y + blockIdx.y*blockDim.y;

	// These are the location of the pixel
	int imgX = globalIdx/direction;
	int imgY = globalIdy;

	__shared__ int counter[5*172]; // Count the nearest distance
	__shared__ float locker[5*172];// Record the none-zero pixel value in each direction
	
	// Get the pixel value of this thread
	float nowPixel = originalImg[imgX + imgY*picW]; 
 	
 	// For non-zero pixels, which are not holes, keep the original value
 	if(nowPixel!= 0)
 	{
 		depthFill[imgX + imgY*picW] = nowPixel; 
 	}

 	else
 	// For all the holes( black pixels), searching in four directions for the nearest non-zero pixel 
 	{
 		// Find which direction this thread is responsible for by doing mod operation
		int arrow = fmodf(globalIdx, direction);

		if(arrow == 0)// Left
		{
			float leftNonzero;
        	int i_left = 1;
	    	bool Got = false;

            while(imgX - i_left >= 0)// Make sure the pixel is within the range of the depth image
            {
            	// Get the pixel value 
	    		leftNonzero = originalImg[imgX-i_left + imgY*picW];

	    		// If the pixel value is none zero, record the distance and pixel value
	    		// Then, jump out of the loop
	    		if(leftNonzero!=0)
	    		{
            		counter[threadIdx.x + threadIdx.y*blockDim.x] = i_left;
	    			locker[threadIdx.x + threadIdx.y*blockDim.x] = leftNonzero;	
	       	 		Got = true;
	      			break;
	    		}

	    		i_left++;

	 		}// End of while, arrow 0
	  	
	  		if(!Got) // If there is no none-zero pixel in the direction, set the counter to a relative large number
	  		{
	    		counter[threadIdx.x + threadIdx.y*blockDim.x] = 400;
	    		locker[threadIdx.x + threadIdx.y*blockDim.x] = 170.0;
	  		}
		}
        
		else if(arrow == 1)// Up
		{
          	float upNonzero;
         	int j_up = 1;
	  		bool Got = false;

          	while(imgY - j_up >= 0)// Make sure the pixel is within the range of the depth image
          	{
	    		// Get the pixel value 
	    		upNonzero = originalImg[imgX + (imgY-j_up)*picW];

	    		// If the pixel value is none zero, record the distance and pixel value
	    		// Then, jump out of the loop
	    		if(upNonzero!=0)
	    		{
              		counter[threadIdx.x + threadIdx.y*blockDim.x] = j_up;
	      			locker[threadIdx.x + threadIdx.y*blockDim.x] = upNonzero;	
	      			Got = true;
	      			break;
	    		}
	    		
	    		j_up++;

	  		}// End of while, arrow 1
	  		
	  		if(!Got)// If there is no none-zero pixel in the direction, set the counter to a relative large number
	  		{
	    		counter[threadIdx.x + threadIdx.y*blockDim.x] = 400;
	    		locker[threadIdx.x + threadIdx.y*blockDim.x] = 170.0;
	  		}
		}

		else if(arrow == 2) //Right
		{
          	float rightNonzero;
          	int i_right = 1;
	  		bool Got = false;
          	
          	while(imgX+i_right<picW)// Make sure the pixel is within the range of the depth image
          	{
	    		rightNonzero = originalImg[imgX+i_right+ imgY*picW];

	    		// If the pixel value is none zero, record the distance and pixel value
	    		// Then, jump out of the loop
	    		if(rightNonzero!=0)
	    		{
              		counter[threadIdx.x + threadIdx.y*blockDim.x] = i_right;
	      			locker[threadIdx.x + threadIdx.y*blockDim.x] = rightNonzero;	
	      			Got = true;
	      			break;
	    		}

	    		i_right++;

	  		}// End of while, arrow 2
	  	
	  		if(!Got)// If there is no none-zero pixel in the direction, set the counter to a relative large number
	  		{
	    		counter[threadIdx.x + threadIdx.y*blockDim.x] = 400;
	    		locker[threadIdx.x + threadIdx.y*blockDim.x] = 170.0;
	  		}
		}

		else // arrow = 3, Down
		{
    		float downNonzero;
    		int j_down = 1;
	  		bool Got = false;
          	
        	while(imgY+j_down<picH)// Make sure the pixel is within the range of the depth image
        	{
	    		downNonzero = originalImg[imgX + (imgY+j_down)*picW];

	    		// If the pixel value is none zero, record the distance and pixel value
	    		// Then, jump out of the loop
	    		if(downNonzero!=0)
	    		{
            		counter[threadIdx.x + threadIdx.y*blockDim.x] = j_down;
	      			locker[threadIdx.x + threadIdx.y*blockDim.x] = downNonzero;	
	      			Got = true;
	      			break;
	    		}
	    		
	    		j_down++;

	  		}// End of while, arrow 3
	  
	  		if(!Got)// If there is no none-zero pixel in the direction, set the counter to a relative large number
	  		{
	    		counter[threadIdx.x + threadIdx.y*blockDim.x] = 400;
	    		locker[threadIdx.x + threadIdx.y*blockDim.x] = 170.0;
	  		}
		}
     
    	// Fill the holes, handled by the left direction thread of each pixel
    	if(threadIdx.x % direction == 0)
    	{
       		int minIndex_x = threadIdx.x;
       		int minValue = counter[threadIdx.x + threadIdx.y*blockDim.x];
       	
       		// These four if statements are walking throught the values in the counter to find the nearest non-zero pixel
       		if(counter[threadIdx.x+1 + threadIdx.y*blockDim.x] < minValue)
       		{
        	 	minIndex_x = threadIdx.x+1;
        	 	minValue = counter[threadIdx.x+1 + threadIdx.y*blockDim.x];
       		}
   
       		if(counter[threadIdx.x+2 + threadIdx.y*blockDim.x] < minValue)
       		{
        	 	minIndex_x = threadIdx.x+2;
        	 	minValue = counter[threadIdx.x+2 + threadIdx.y*blockDim.x];
       		}

       		if(counter[threadIdx.x+3 + threadIdx.y*blockDim.x] < minValue)
       		{
        	 	minIndex_x = threadIdx.x+3;
        	 	//minValue = counter[threadIdx.x+2 + threadIdx.y*blockDim.x];
       		}
     	
     		// Fill the hole with the value of the nearest non-zero pixel
     		depthFill[imgX + imgY * picW] = locker[minIndex_x + threadIdx.y*blockDim.x];
    
    	} // End of filling hole
       
    } // End of else, for pixel = 0

}//getDepthFill


__global__ void maskDistance(float* mask_distance, float variationD)
{
	// This function calculates the spatial Gaussian, 
	// As the mask size is constant, the spatial Gaussian is invariant from pixel to pixel

	__shared__ float mask_loc[maskSize*maskSize];
 	int tx = threadIdx.x;
 	int ty = threadIdx.y;

	// Generate distance mask by using Gaussian distribution:
	mask_loc[ty*maskSize+tx] = ( 1.0/(2*PI*powf(variationD,2))) * expf(-1.0*(powf((tx-boundary),2)+powf((ty-boundary),2)) / (2.0*powf(variationD,2)));

	// Transfer the result back
	mask_distance[ty*maskSize+tx] = mask_loc[ty*maskSize+tx];
}

// Extend the image with patches with size maskSize*maskSize
// Every patch is filled with the value of corresponding pixel value of original image
__global__ void extend(float* inImg, float* outImg)
{
	int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int globalIdy = threadIdx.y + blockIdx.y*blockDim.y;
	int inImgx = globalIdx/maskSize + globalIdx % maskSize - boundary;
	int inImgy = globalIdy/maskSize + globalIdy % maskSize - boundary;


	// Get the extended matrix, the extended matrix can be divided into picH x picW blocks
	// Each block is of size masksize x masksize
	// The centeral element of each block is the (blockNumIdx,blockNumIdy) element of the original image
	// The surrounding pixels are the corresponding pixels
	// If the surrounding is out of the original image, the element is assigned to be zero
	if(inImgx>=0 && inImgx<picW && inImgy>=0 && inImgy<picH)
	{ 
		outImg[globalIdx + globalIdy*eImgW] = inImg[inImgx + inImgy * picW];
	}
	else
	{
		outImg[globalIdx + globalIdy*eImgW] = 0.0;
	}

}//extend func

// Calculate the range Gaussian for every element of every mask block for every pixel in the original image
// For those elelments who lie outside of the bounday, assigned their values to be zero
__global__ void intenseFunc(const float* __restrict__ extendedImg, float* intenseMat, float variationI)
{
	int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int globalIdy = threadIdx.y + blockIdx.y*blockDim.y; 
	int focusx = (globalIdx/maskSize)*maskSize + boundary;
	int focusy = (globalIdy/maskSize)*maskSize + boundary;

	if(extendedImg[globalIdx + globalIdy * eImgW] == 0)// Elements outside the boundary
	{
		intenseMat[globalIdx + globalIdy * eImgW]=0.0;
	}
	else // Elements within the range, calculate the range Gaussian 
	{
 		intenseMat[globalIdx + globalIdy * eImgW] = (1.0/(2.0*PI*powf(variationI, 2))) * expf( -1.0*powf((extendedImg[globalIdx + globalIdy * eImgW] - extendedImg[focusx + focusy * eImgW] ),2)/(2*powf(variationI,2)) );
	}

}//intenseFunc 

// Calculate the weight value and the product of the intensity with the product of range Gaussian and spatial Gaussian for each element in the extended matrix
__global__ void numAndden(const float* __restrict__ maskDistance, const float* __restrict__ maskIntense, const float* __restrict__ extendedImg, float* numMat, float* denMat)
{
	int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int globalIdy = threadIdx.y + blockIdx.y*blockDim.y;

	// The sum of range Gaussian and spatial Gaussian for each pixel in the original image( or we can say for each mask block)
	float tmp=0.0;
	tmp = maskDistance[globalIdx % maskSize + (globalIdy % maskSize)*maskSize] * maskIntense[globalIdx + globalIdy * eImgW];

	denMat[globalIdx + globalIdy * eImgW] = tmp; // Weight
	numMat[globalIdx + globalIdy * eImgW] = tmp * extendedImg[globalIdx + globalIdy * eImgW];// Product of intensity with the sum
}

// Sum up for each block and get the final value of the result image
__global__ void bilateral(const float* __restrict__ numMat, const float* __restrict__ denMat, float* resMat)
{
	int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
	int globalIdy = threadIdx.y + blockIdx.y*blockDim.y;
	float numValue = 0.0;
	float denWeight = 0.0;
 	for(int j=globalIdy*maskSize; j<(globalIdy+1)*maskSize; j++)
 	{
  		for(int i=globalIdx*maskSize; i<(globalIdx+1)*maskSize; i++)
  		{
   			numValue+=numMat[i+j*eImgW];
   			denWeight+=denMat[i+j*eImgW];
  		}
 	}
  	
  	if(denWeight != 0)
  	{
  		resMat[globalIdx + globalIdy*picW] = numValue/denWeight;
  	}
  	else 
  	{
  		resMat[globalIdx + globalIdy*picW] = 170;
  	}

}
"""


# Read the depth image
def create_img(filename, cols, rows):
	size = (cols, rows)
	im = Image.open(filename).convert('L')
        im = im.resize(size)
        return np.array(im)

# Variables:

# Size of depth image
picH = 300
picW = 430

direction = 4
# Read in the depth image
depthImg = create_img("./Depth.jpg", picW, picH)
depthImg = np.float32(depthImg)

# Pre-allocate array for hole filling result and filtering result
depthFill = np.zeros((picH, picW)).astype(np.float32)
denDepth = np.zeros((picH, picW)).astype(np.float32)



# Read brightness value of depthPicture from '/home/ftqtf/Documents/Processing/project/data/depthData.txt'
originaltxt = np.loadtxt("./depthData.txt", dtype=np.float32, delimiter=",")
# txtMatrix = txtMatrix.transpose()
txtMatrix = np.zeros((picH, picW)).astype(np.float32)

# Manually transpose the matrix for usage in the gpu
for i in range(picW):
	for j in range(picH):
		txtMatrix[j,i] = originaltxt[i,j]


# Set some constant 

# Mask size for bilateral filter
maskSize = np.int32(9)
boundary = maskSize/2

# Preallocate array for intermediate quantity and result of bilateral filter
mask_distance = np.zeros((maskSize, maskSize)).astype(np.float32)
mask_intensity = np.zeros((maskSize, maskSize)).astype(np.float32)
mask_bilateral = np.zeros((maskSize, maskSize)).astype(np.float32)

# Variations for spatial Gaussian and range Gaussian for bilateral filter
variationD = 2.0   #distance
variationI = 25.0  #intensity



##################################################
## Python Implementation Starts Here
##################################################

start_py = time.time()

# This for loop is doing hole filling by searching for the nearest non-black pixel for every black pixel in the depth image
# and use the value of the nearest non-black pixel to fill the black pixel 
for i in range(picW):
	for j in range(picH):
    		leftNonzero = 170.0  #grayscale value
    		rightNonzero = 170.0
    		upNonzero = 170.0
    		downNonzero = 170.0

    	# For all black pixel, doing the hole filling
   		if txtMatrix[j][i] == 0.0:
      			#fillValue
      			fillValue = 0.0
			
			# Left
			i_left = 1
			while i - i_left >= 0: # 
				leftV = txtMatrix[j][i - i_left]
				if leftV != 0.0:
					leftNonzero = leftV
					break
				i_left += 1

			# Right
			i_right = 1
			while i + i_right < picW:
				rightV = txtMatrix[j][i+i_right]
				if rightV != 0.0:
					rightNonzero = rightV
					break
				i_right += 1
	
			# Up
			j_up = 1
			while j - j_up >= 0:
				upV = txtMatrix[j-j_up][i]
				if upV != 0.0:
					upNonzero = upV
					break
				j_up += 1

			# Down
			j_down=1
			while j+j_down<picH:
				downV = txtMatrix[j+j_down][i]
				if downV != 0.0:
					downNonzero = downV
					break
				j_down+=1

			# Find the nearest one
			min1 = min(i_left, i_right)
			min2 = min(j_up, j_down)
			minFinal = min(min1, min2)

			# Fill the hole with the nearest one 
			if minFinal == i_left:
				fillValue = leftNonzero
			elif minFinal == i_right:
				fillValue = rightNonzero
			elif minFinal == j_up:
				fillValue = upNonzero
			else:
				fillValue = downNonzero

			depthFill[j][i] = fillValue
		
		# Filling the non-black pixel with original value
		else :
			depthFill[j][i] = txtMatrix[j][i]


# Generate Spatial Gaussian for bilateral filter
for i in range(-boundary, boundary+1, 1):
	for j in range(-boundary, boundary+1, 1):
        	mask_distance[i+boundary][j+boundary] = (1.0/(2.0*np.pi*pow(variationD,2))) * np.exp( -1.0*(pow(i,2)+pow(j,2))/(2.0*pow(variationD,2)) )

# Bilateral filter
for i in range(picH):
  for j in range(picW):

    sumWeight = 0.0

    for innerI in range(-boundary+i, boundary+i+1, 1):
    	for innerJ in range(-boundary+j, boundary+j+1, 1):
			if innerI>=0 and innerI<picH and innerJ>=0 and innerJ<picW:
				mask_intensity[innerI+boundary-i][innerJ+boundary-j] = (1.0/(2.0*np.pi*pow(variationI,2))) * np.exp(-pow((depthFill[i][j] - depthFill[innerI][innerJ]),2) / (2*pow(variationI,2)) )
			else:
				mask_intensity[innerI+boundary-i][innerJ+boundary-j] = 0.0
			mask_bilateral[innerI+boundary-i][innerJ+boundary-j] = mask_intensity[innerI+boundary-i][innerJ+boundary-j] * mask_distance[innerI+boundary-i][innerJ+boundary-j]
			sumWeight +=mask_bilateral[innerI+boundary-i][innerJ+boundary-j]
	
    ResValue = 0.0
    for innerI in range(-boundary+i, boundary+i+1, 1):
      	for innerJ in range(-boundary+j, boundary+j+1, 1):
       		if innerI>=0 and innerI<picH and innerJ>=0 and innerJ<picW:
				ResValue+=mask_bilateral[innerI+boundary-i][innerJ+boundary-j]*depthFill[innerI][innerJ]
        	else:
	  			ResValue+=0.0
    		denDepth[i][j] = ResValue/sumWeight 

finish_py = time.time()

print 'Python execution time: ', finish_py - start_py
##################################################
## CUDA Implementation Starts Here
##################################################

# Size of the extended Img:
eImgH = picH * maskSize
eImgW = picW * maskSize

# Transfer parameter from host to kernel

# Pre-allocate memory space in kernel to hold intermediate result
depthFillK = np.zeros((picH, picW)).astype(np.float32)
mask_distanceK = np.zeros_like(mask_distance)
mask_intensityK = np.zeros((eImgH, eImgW)).astype(np.float32)
mask_extendedImgK = np.zeros((eImgH, eImgW)).astype(np.float32)
mask_bilateralK = np.zeros((picH, picW)).astype(np.float32)
numMatK = np.zeros((eImgH, eImgW)).astype(np.float32)
denMatK = np.zeros((eImgH, eImgW)).astype(np.float32)

distance_gpu = gpuarray.to_gpu(mask_distanceK)
intensity_gpu = gpuarray.to_gpu(mask_intensityK)
extend_gpu = gpuarray.to_gpu(mask_extendedImgK)
bilateral_gpu = gpuarray.to_gpu(mask_bilateralK)
num_gpu = gpuarray.to_gpu(numMatK)
den_gpu = gpuarray.to_gpu(denMatK)
original_gpu = gpuarray.to_gpu(txtMatrix)
depthFill_gpu = gpuarray.to_gpu(depthFillK)

# Get the kernel functions
mod = compiler.SourceModule(kernel_code)
maskDistance = mod.get_function("maskDistance")
extend = mod.get_function("extend")
intenseFunc = mod.get_function("intenseFunc")
numAndden = mod.get_function("numAndden")
bilateral = mod.get_function("bilateral")
getDepthFill = mod.get_function("getDepthFill")

# Lists for plotting
testBlockSizeX = [2,2,5,5,5,5,5,5,10,10,10,43,43,43,43]
testBlockSizeY = [2,3,2,3,5,6,10,15,10,15,25,6,10,15,20]
totalBlockSize = [] 
timeCuda = []

for indexBlock in range(len(testBlockSizeX)):
	blockSizeX = testBlockSizeX[indexBlock]
	blockSizeY = testBlockSizeY[indexBlock]
	gridX = picW*maskSize/blockSizeX
	gridY = picH*maskSize/blockSizeY
	totalBlockSize.append(blockSizeX*blockSizeY)
	#start time
	start = time.time()

	#get depthFill
	getDepthFill(original_gpu, depthFill_gpu, grid=(10,60), block=(direction*picW/10, picH/60, 1),)

	#get distance matrix
	maskDistance(distance_gpu, np.float32(variationD), grid=(1,1), block=(int(maskSize),int(maskSize),1),)

	#extend depthFill image
	#depthFill_gpu = gpuarray.to_gpu(np.float32(depthFill)) #then this one can be changed
	extend(depthFill_gpu, extend_gpu, grid=(gridX, gridY), block=(int(blockSizeX),int(blockSizeY), 1),)

	#get intensity
	intenseFunc(extend_gpu, intensity_gpu, np.float32(variationI), grid=(gridX, gridY), block=(blockSizeX, blockSizeY, 1),)

	#get num and den
	numAndden(distance_gpu, intensity_gpu, extend_gpu, num_gpu, den_gpu, grid=(gridX, gridY), block=(blockSizeX, blockSizeY, 1),)

	#get bilateral result
	bilateral(num_gpu, den_gpu, bilateral_gpu, grid = (gridX/maskSize, gridY/maskSize), block = (blockSizeX, blockSizeY, 1),)

	#end time
	end = time.time()
	timeSpent = end - start
	timeCuda.append(timeSpent)
	print str(timeSpent)+'s'

# Draw the python results
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(depthImg, cmap='Greys_r')

plt.subplot(1,3,2)
plt.title("Hole Filled Python")
plt.imshow(depthFill, cmap='Greys_r')

plt.subplot(1,3,3)
plt.title("Bilateral Filter Result Python")
plt.imshow(denDepth, cmap='Greys_r')

# Save python result
plt.savefig('Kine_result_py.png')
plt.gcf().clear()

# Draw the cuda result
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(original_gpu.get(), cmap='gray')

plt.subplot(1,3,2)
plt.title("Hole Filled CUDA")
plt.imshow(depthFill_gpu.get(), cmap='gray')

plt.subplot(1,3,3)
plt.title("Bilateral Filter Result CUDA")
plt.imshow(bilateral_gpu.get(), cmap='gray')

# Save cuda result
plt.savefig('Kine_result_cuda.png')
plt.gcf().clear()

plt.imshow(original_gpu.get(), cmap='gray')
plt.savefig('Kine_result_original.png')
plt.gcf().clear()

plt.imshow(depthFill_gpu.get(), cmap='gray')
plt.savefig('Kine_result_holefilled.png')
plt.gcf().clear()

plt.imshow(bilateral_gpu.get(), cmap='gray')
plt.savefig('Kine_result_filtered.png')
plt.gcf().clear()

# Draw timing result
plt.plot(totalBlockSize, timeCuda, 'r')
plt.legend(['time CUDA'], loc='upper left')
plt.title("time needed for holes filling and bilateral filtering")
plt.xlabel('totalBlockSize (X*Y)')
plt.ylabel('time /sec')
plt.gca().set_xlim((min(totalBlockSize), max(totalBlockSize)))
#plt.imshow()

# Save timing result
plt.savefig('Timing_result.png')
