import numpy as np
from keras.utils import Sequence
from PIL import Image

class ImageDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, image_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        batch_images = []
        batch_labels = []
        for i in batch_indices:
            img = Image.open(self.image_paths[i])
            img = img.resize(self.image_size)  # Resize image if necessary
            img_array = np.array(img) / 255.0  # Normalize image values
            batch_images.append(img_array)
            batch_labels.append(self.labels[i])
        
        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        # Optional: shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)
