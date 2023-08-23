import cv2
import numpy as np
import matplotlib.pyplot as plt

def piecewise_linear_transform(image, breakpoints, slopes):
    
    # Create a lookup table for the transformation
    lookup_table = np.zeros(256, dtype=np.uint8)
    
    for i in range(len(breakpoints) - 1):
        start, end = breakpoints[i], breakpoints[i+1]
        slope = slopes[i]
        lookup_table[start:end] = np.clip(np.arange(start, end) * slope, 0, 255)
    
    # Apply the transformation using the lookup table
    transformed_image = cv2.LUT(image, lookup_table)
    return transformed_image

# Load an example image
image_path = "D:\downloads\emma.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the piecewise transformation parameters
breakpoints = [0, 50, 150, 255]
slopes = [1, 1.55, 1]

# Apply the piecewise linear transformation
transformed_image = piecewise_linear_transform(original_image, breakpoints, slopes)

# Display the original and transformed images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title("Transformed Image")

plt.tight_layout()
plt.show()
