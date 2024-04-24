import tensorflow as tf
import numpy as np
import pandas as pd
import os
import nibabel as nib
from scipy.ndimage import zoom, rotate
from tensorflow.image import adjust_brightness

class DatasetManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_images_and_labels(self, csv_name, folder_name):
        csv_path = os.path.join(self.base_dir, csv_name)
        images_folder_path = os.path.join(self.base_dir, folder_name)
        
        data = pd.read_csv(csv_path)
        data['image_path'] = data['Biosample'].apply(lambda x: os.path.join(images_folder_path, f"{x}.mnc"))
        data = data[data['image_path'].apply(os.path.exists)]
        
        if data.empty:
            print("No image files found.")
            return [], []
        
        data['label'] = data['Experimental_Group'].apply(lambda x: 0 if x == 'Control' else 1)
        return data['image_path'].tolist(), data['label'].tolist()

    def preprocess_and_augment_image(self, file_path, label, augment_type):
        image = tf.py_function(func=self.load_process_and_augument_image, inp=[file_path, augment_type], Tout=tf.float32)
        image.set_shape((256, 256, 256, 1))
        return image, label

    def load_process_and_augument_image(self, file_path, augment_type):
        try:
            # Load image data
            image = nib.load(file_path.numpy().decode())
            image_data = image.get_fdata()

            # Downsample and normalize
            processed_image = self.downsample_and_normalize_image(image_data)

            # Conditional augmentation
            if augment_type.numpy().decode() == 'rotate':
                processed_image = self.augment_image_rotate(processed_image)
            elif augment_type.numpy().decode() == 'brightness':
                processed_image = self.augment_image_brightness(processed_image)

            return processed_image
        except Exception as e:
            print(f"Failed to process file {file_path.numpy().decode()}: {str(e)}")
            return np.zeros((256, 256, 256, 1), dtype=np.float32)

    def downsample_and_normalize_image(self, image_data, target_shape=(256, 256, 256)):
        scale_factors = (target_shape[0] / image_data.shape[0], 
                         target_shape[1] / image_data.shape[1], 
                         target_shape[2] / image_data.shape[2])
        resized_image = zoom(image_data, scale_factors, order=1)
        normalized_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))
        normalized_image = normalized_image[..., np.newaxis]
        return normalized_image.astype(np.float32)

    def augment_image_rotate(self, image_data, angle=10):
        return rotate(image_data, angle, axes=(0, 1), reshape=False, mode='nearest')

    def augment_image_brightness(self, image_data, delta=0.1):
        image_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
        brightened_image = adjust_brightness(image_tensor, delta)
        return brightened_image.numpy()

    def prepare_dataset(self, image_paths, labels, batch_size, shuffle=False, augment_type=None):
        augment_type = tf.constant(augment_type if augment_type else '')
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: self.preprocess_and_augment_image(x, y, augment_type), num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(labels))
        dataset = dataset.batch(batch_size)
        return dataset
