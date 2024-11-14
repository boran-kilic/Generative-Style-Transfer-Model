import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread(r'C:\Generative-Style-Transfer-Model\test\content_test\000000052413.jpg')

# Check if the image is loaded
if img is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Convert BGR to RGB (OpenCV loads images in BGR format by default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image dimensions
h, w, c = img.shape  # For example: (768, 1024, 3)

# Generate random noise
noise = np.random.randint(0, 50, (h, w), dtype=np.uint8)  # Design jitter/noise here
zitter = np.zeros_like(img, dtype=np.uint8)  # Create an empty array of the same shape as img
zitter[:, :, 0] = noise
zitter[:, :, 1] = noise
zitter[:, :, 2] = noise  

# Add noise to the image
noise_added = cv2.add(img, zitter)

# Combine original and noisy images
combined = np.vstack((img[:h // 2, :, :], noise_added[h // 2:, :, :]))  # Use integer division //

# Display the combined image
plt.imshow(combined, interpolation='none')
plt.axis('off')  # Turn off axis for better visualization
plt.show()
