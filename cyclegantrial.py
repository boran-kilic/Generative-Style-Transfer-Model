import tensorflow as tf
import os
from tensorflow_examples.models.pix2pix import pix2pix
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE

# Define paths
dataset_dir = 'dataset'
trainA_path = os.path.join(dataset_dir, 'train/content_train')
trainB_path = os.path.join(dataset_dir, 'train/style_train')
testA_path = os.path.join(dataset_dir, 'test/content_test')
testB_path = os.path.join(dataset_dir, 'test/style_test')

# Function to load and preprocess images
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [286, 286])


    image = (image / 127.5) - 1  # Normalize to [-1, 1] 
    return image

# Function to load dataset from a directory
def load_dataset(directory):
    dataset = tf.data.Dataset.list_files(os.path.join(directory, '*.jpg'))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Load datasets
trainA = load_dataset(trainA_path)
trainB = load_dataset(trainB_path)
testA = load_dataset(testA_path)
testB = load_dataset(testB_path)

# Combine into a dictionary
dataset = {
    'trainA': trainA,
    'trainB': trainB,
    'testA': testA,
    'testB': testB
}

train_content, train_style = dataset['trainA'], dataset['trainB']
test_content, test_style = dataset['testA'], dataset['testB']


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
#   image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image


def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = normalize(image)
  return image



train_content = train_content.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_style = train_style.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_content = test_content.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_style = test_style.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)
    
sample_content = next(iter(train_content))
sample_style = next(iter(train_style))    

print("Sample content shape:", sample_content[0].shape)
print("Sample content min/max:", tf.reduce_min(sample_content[0]).numpy(), tf.reduce_max(sample_content[0]).numpy())
print("Sample style shape:", sample_style[0].shape)
print("Sample style min/max:", tf.reduce_min(sample_style[0]).numpy(), tf.reduce_max(sample_style[0]).numpy())

plt.subplot(121)
plt.title('Content')
plt.imshow(tf.cast(sample_content[0] * 0.5 + 0.5, tf.float32))

plt.subplot(122)
plt.title('Content with random jitter')
plt.imshow(tf.cast(random_jitter(sample_content[0]) * 0.5 + 0.5, tf.float32))

plt.figure()

plt.subplot(121)
plt.title('Style')
plt.imshow(tf.cast(sample_style[0] * 0.5 + 0.5, tf.float32))

plt.subplot(122)
plt.title('Style with random jitter')
plt.imshow(tf.cast(random_jitter(sample_style[0]) * 0.5 + 0.5, tf.float32))

plt.show()

    