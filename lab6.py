import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as nd

#################### Part 1 ############################
img = cv2.imread('lab6.bmp', 0)
img = img.astype(np.float32)
h, w = img.shape
volume = np.zeros((h,w,3),np.float32)

def get_kernel(sigma): # function to obtain the kernel
	kernel = int(2 * round(3 * sigma) + 1)
	return kernel

sigma = 2
kernel = get_kernel(sigma)
dst = cv2.GaussianBlur(img, (kernel,kernel), sigma, sigma, 0)

plt.subplot(211),plt.imshow(img),plt.title('Input Image')
plt.subplot(212),plt.imshow(dst),plt.title('Blurred Image')
plt.show()
########################################################

sigma1 = 3
sigma2 = 4
sigma3 = 5

kernel1 = get_kernel(sigma1)
dst1 = cv2.GaussianBlur(dst, (kernel1,kernel1), sigma1, sigma1, 0)
#cv2.Laplacian(src,ddepth,ksize = kernel_size,scale = scale,delta = delta)
dst_1 = cv2.Laplacian(dst1, ddepth = -1, ksize = kernel1, scale = 1, delta = 0)

kernel2 = get_kernel(sigma2)
dst2 = cv2.GaussianBlur(dst, (kernel2,kernel2), sigma2, sigma2, 0)
dst_2 = cv2.Laplacian(dst2, ddepth = -1, ksize = kernel2, scale = 1, delta = 0)

kernel3 = get_kernel(sigma3)
dst3 = cv2.GaussianBlur(dst, (kernel3,kernel3), sigma3, sigma3, 0)
dst_3 = cv2.Laplacian(dst3, ddepth = -1, ksize = kernel3, scale = 1, delta = 0)

# merge the channels in volume
volume[:,:,0]=dst_1
volume[:,:,1]=dst_2
volume[:,:,2]=dst_3

plt.subplot(322),plt.imshow(dst_1),plt.title('Level 1')
plt.subplot(324),plt.imshow(dst_2),plt.title('Level 2')
plt.subplot(326),plt.imshow(dst_3),plt.title('Level 3')
plt.show()

#################### Part 2 ############################

local_max = nd.filters.minimum_filter(volume, size=3, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
msk = (volume == local_max)
output_image = np.sum(msk, axis = 2,dtype = np.int32, out = None, keepdims = False)

non_zero_x, non_zero_y = np.nonzero(output_image)
#plt.subplot(111),plt.imshow(output_image),plt.title('mask')

plt.subplot(111),plt.imshow(img),plt.title('Rough blobs detected in the image')
plt.scatter(non_zero_y, non_zero_x, c='red')
plt.xlim([0,300])
plt.ylim([0,140])
plt.show()

#################### Part 3 ############################
dst = np.array(dst, np.uint8) # convert the gaussian blurred image obtained in 
#step 2 to uint8, prevents src.type() == CV_8UC1 error in cv2.threshold
ret_value,threshold = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret_value = the optimal threshold  

#height, width = np.shape(img) # get the height and width of the 
# original image or the output_image
height, width = np.shape(output_image)

for a in range(height):
        for b in range(width):
                if dst[a][b] <= ret_value: # if the gaussian blurred image
                        # at x,y (i.e. a,b) is less than the optimal threshold
                        # + 1, value of pixel at that location is 0
                        output_image[a][b] = 0

non_zero_x1, non_zero_y1 = np.nonzero(output_image) # getting rid of zeros

plt.subplot(111),plt.imshow(img),plt.title('Refined blobs detected in the image')
plt.scatter(non_zero_y1, non_zero_x1, c='red')
plt.xlim([0,300])
plt.ylim([0,140])
plt.show()
