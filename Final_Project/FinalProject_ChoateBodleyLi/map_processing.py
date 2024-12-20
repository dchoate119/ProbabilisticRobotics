# Daniel Choate, Reed Bodley, Yifan Li

# Post processing GMapping results to smooth map lines and eliminate noise 

import cv2
import numpy as np 

# Import old map 
# map_image = cv2.imread("jcc_first_floor.pgm", cv2.IMREAD_GRAYSCALE)
map_image = cv2.imread("computer_lab.pgm", cv2.IMREAD_GRAYSCALE)
# map_image = cv2.imread("test_map.pgm", cv2.IMREAD_GRAYSCALE)
print(map_image.shape)


# Specify the desired dimensions to crop specific pictures 
# COMPUTER LAB MAP 
width_s = 100
width_e = width_s + 225 # desired width in pixels
height_s = 140
height_e = height_s + 225  # desired height in pixels

# # JCC FIRST FLOOR 
# width_s = 0
# width_e = width_s + 1250 # desired width in pixels
# height_s = 0
# height_e = height_s + 1250  # desired height in pixels

# # JCC THIRD FLOOR 
# width_s = 0
# width_e = width_s + 1250 # desired width in pixels
# height_s = 0
# height_e = height_s + 1250  # desired height in pixels


print('The selected pixel ratio is', width_e-width_s, 'x', height_e-height_s)

# Crop the image 
im_crop = map_image[height_s:height_e, width_s:width_e] #, height_s:height_e]
# print(im_crop.size)
# print(im_crop.shape)

# Resize the image 
width_n = 400
height_n = 400
new_dim = (width_n, height_n)
im_res = cv2.resize(im_crop, new_dim)
# print(im_res.shape)



# Apply Gaussian blur to smooth edges
blurred_map = cv2.GaussianBlur(im_res, (3,3), 0)

# cv2.imshow('Blurred Map', blurred_map)
# cv2.imshow('Original Map', im_res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Apply bilateral filter after Gaussian blur
smooth_image = cv2.bilateralFilter(blurred_map, 9, 75, 75)

# cv2.imshow('Smoothed Image', smooth_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Optional: Apply thresholding to make edges sharper
_, thresholded_map = cv2.threshold(smooth_image, 128, 255, cv2.THRESH_BINARY)
# _, thresholded_map = cv2.threshold(im_res, 128, 255, cv2.THRESH_BINARY)

# Optional: Dilate to fill small gaps
kernel = np.ones((2, 2), np.uint8)
dilated_map = cv2.dilate(thresholded_map, kernel, iterations=1)


cv2.imshow('Original Map', im_res)
cv2.imshow('Blurred Map', blurred_map)
cv2.imshow('Smoothed Image', smooth_image)
cv2.imshow('Thresholded and Dilated Map', dilated_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
