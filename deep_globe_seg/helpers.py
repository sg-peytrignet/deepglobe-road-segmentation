import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A
import logging
from tensorflow import keras
from typing import Dict, Tuple
import tensorflow as tf

def visualize(**images):
    """
    Plot images in one row.

    Args:
        **images: Arbitrary keyword arguments containing image arrays. 
                  The keys are used as titles for the images.

    Returns:
        None. This function displays a plot of the images.
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


def round_clip_0_1(x, **kwargs):
    """
    Round the input array and clip its values to the range 0-1.

    Args:
        x (numpy.ndarray): The input array to round and clip.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        numpy.ndarray: The rounded and clipped array.
    """
    return x.round().clip(0, 1)


def denormalize(x):
    """
    Scale image array to the range 0..1 for correct plotting.

    Args:
        x (numpy.ndarray): The input image array to be denormalized.

    Returns:
        numpy.ndarray: The scaled image array with values in the range 0..1.
    """
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def list_subfolders_and_files(base_folder, num_files=10):
    """
    List all subfolders and a customizable number of files in each subfolder of the given base folder.

    Args:
        base_folder (str): The base folder path to list subfolders and files from.
        num_files (int, optional): Number of files to list per subfolder. Defaults to 10.

    Returns:
        None
    """
    # List all subfolders
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    for subfolder in subfolders:
        print(f"Subfolder: {subfolder}")
        
        # List all files in the subfolder, sorted alphabetically
        files = sorted([f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))])
        
        # Get the specified number of files
        num_files_to_list = min(num_files, len(files))
        files_to_display = files[:num_files_to_list]
        
        for file in files_to_display:
            print(f"  {file}")


class Path:
    def __init__(self, 
                 model_type: str,
                 image_folder: str,
                 metadata: str,
                 class_dict: str,
                 train: str,
                 valid: str,
                 test: str,
                 tensorboard_logs_path: str,
                 saved_model_folder: str,
                 model_save_path: str):
        self.model_type = model_type
        self.image_folder = image_folder
        self.metadata = metadata
        self.class_dict = class_dict
        self.train = train
        self.valid = valid
        self.test = test
        self.tensorboard_logs_path = tensorboard_logs_path
        self.saved_model_folder = saved_model_folder
        self.model_save_path = model_save_path


class Inputs:
    def __init__(self, img_size=(512, 512), input_shape=(512, 512, 3)):
        self.img_size = img_size
        self.input_shape = input_shape


class HyperParameter:
    def __init__(self, 
                 batch_size: int = 16, 
                 learning_rate: float = 0.00005, 
                 num_classes: int = 1, 
                 epochs: int = 40):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.epochs = epochs


class CFG:
    """
    Configuration class for defining paths, dataset parameters, and hyperparameters
    for a machine learning model.
    """
    def __init__(
        self,
        image_folder: str,
        saved_model_folder: str,
        tensorboard_logs_path: str,
        dataset_params: Dict,
        hyper_params: Dict,
        model_type: str
    ):
        self.model_type = model_type
        # Path definition
        self.Path = Path(
            model_type=model_type,
            image_folder=image_folder,
            metadata=os.path.join(image_folder, 'metadata.csv'),
            class_dict=os.path.join(image_folder, 'class_dict.csv'),
            train=os.path.join(image_folder, 'train/'),
            valid=os.path.join(image_folder, 'valid/'),
            test=os.path.join(image_folder, 'test/'),
            tensorboard_logs_path=tensorboard_logs_path,
            saved_model_folder=saved_model_folder,
            model_save_path=f'{saved_model_folder}/{model_type}.weights.h5'
        )
        
        # Dataset parameters
        self.Dataset = Inputs(
            img_size=dataset_params['img_size'],
            input_shape=dataset_params['input_shape']
        )
        
        # Hyperparameters
        self.HyperParameter = HyperParameter(
            batch_size=hyper_params['batch_size'],
            learning_rate=hyper_params['learning_rate'],
            num_classes=hyper_params['num_classes'],
            epochs=hyper_params['epochs']
        )


def get_training_augmentation(size):
    """
    Create a training augmentation pipeline with heavy augmentations.
    
    Args:
        height (int): Target height of the image after augmentation.
        width (int): Target width of the image after augmentation.
        
    Returns:
        albumentations.Compose: A composition of augmentation transformations.
    """
    height = size[0]
    width = size[1]
    train_transform = [
        A.HorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True),  # Pad the image if needed
        A.RandomCrop(height=height, width=width, always_apply=True),  # Randomly crop the image
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1),  # Shift, scale, and rotate the image
        A.RandomBrightnessContrast(p=0.9),  # Apply random brightness and contrast adjustments
        A.Blur(blur_limit=3, p=0.9),  # Apply random blur
        A.HueSaturationValue(p=0.9),  # Apply random hue, saturation, and value adjustments
        A.Lambda(mask=round_clip_0_1)  # Apply a custom transformation to the mask
    ]

    return A.Compose(train_transform)


def get_validation_augmentation(size):
    """
    Create a validation augmentation pipeline with minimal transformations.
    
    Args:
        height (int): Target height of the image after augmentation.
        width (int): Target width of the image after augmentation.
        
    Returns:
        albumentations.Compose: A composition of augmentation transformations.
    """
    height = size[0]
    width = size[1]
    test_transform = [
        A.PadIfNeeded(height, width)  # Pad the image if needed to make the shape divisible by the network input size
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """
    Create a preprocessing pipeline with a specified normalization function.
    
    Args:
        preprocessing_fn (callable): Data normalization function 
            (can be specific for each pretrained neural network)
            
    Returns:
        albumentations.Compose: A composition of preprocessing transformations.
    """
    _transform = [
        A.Lambda(image=preprocessing_fn),  # Apply the specified preprocessing function
    ]
    return A.Compose(_transform)


class Dataset:
    """
    CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        df (pd.DataFrame): DataFrame containing paths to images and masks.
        shape (tuple): Desired shape for the images and masks (height, width).
        classes (list): List of class names to extract from segmentation masks.
        augmentation (albumentations.Compose, optional): Data transformation pipeline (e.g. flip, scale, etc.).
        preprocessing (albumentations.Compose, optional): Data preprocessing pipeline (e.g. normalization, shape manipulation, etc.).
    """
    # Define the default classes
    CLASSES = ['road']
    
    def __init__(self, df, shape, classes=None, augmentation=None, preprocessing=None):
        # Store initialization parameters
        self.df = df
        self.shape = shape
        # If no classes are provided, use the default classes
        self.classes = classes if classes else self.CLASSES
        # Convert class names to their respective indices
        self.class_values = self._get_class_values(self.classes)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        # Validate the provided inputs
        self._validate_inputs()

        # Extract image and mask file paths from the dataframe
        self.images_fps = df['sat_image_path'].tolist()
        self.masks_fps = df['mask_path'].tolist()
        # Store the number of samples
        self.length = len(df)
        
        logging.info(f"Initialized Dataset with {self.length} samples.")
    
    def __len__(self):
        # Return the number of samples
        return self.length
    
    def __getitem__(self, i):
        try:
            # Load and process the image and mask at the given index
            image, mask = self._load_data(i)
            mask = self._process_mask(mask)
            
            # Apply data augmentation if provided
            if self.augmentation:
                image, mask = self._apply_augmentation(image, mask)
            # Apply data preprocessing if provided
            if self.preprocessing:
                image, mask = self._apply_preprocessing(image, mask)
                
            return image, mask
        except Exception as e:
            logging.error(f"Error processing index {i}: {e}")
            raise
    
    def _load_data(self, i):
        # Get the file paths for the image and mask
        image_path = self.images_fps[i]
        mask_path = self.masks_fps[i]

        # Read the image from disk
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        # Convert the image from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask from disk (in grayscale)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Resize the image and mask to the desired shape
        image = cv2.resize(image, self.shape)
        mask = cv2.resize(mask, self.shape)
        
        return image, mask

    def _process_mask(self, mask):
        # Invert the binary mask (swap black and white)
        mask = cv2.bitwise_not(mask)
        
        # Create a list of masks for each class based on the class values
        masks = [(mask == v) for v in self.class_values]
        
        # Stack the masks along a new dimension (channels) to create a multi-class mask
        mask = np.stack(masks, axis=-1).astype('float')
        
        # Add a background channel if the mask is not binary
        if mask.shape[-1] != 1:
            # Create the background channel by subtracting the sum of all class channels from 1
            background = 1 - mask.sum(axis=-1, keepdims=True)
            # Concatenate the background channel to the mask
            mask = np.concatenate((mask, background), axis=-1)
        
        return mask

    def _apply_augmentation(self, image, mask):
        # Apply the augmentation pipeline to the image and mask
        sample = self.augmentation(image=image, mask=mask)
        return sample['image'], sample['mask']
    
    def _apply_preprocessing(self, image, mask):
        # Apply the preprocessing pipeline to the image and mask
        sample = self.preprocessing(image=image, mask=mask)
        return sample['image'], sample['mask']
    
    def _get_class_values(self, classes):
        # Convert class names to their respective indices
        return [self.CLASSES.index(cls.lower()) for cls in classes]
    
    def _validate_inputs(self):
        # Validate the input dataframe
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        # Validate the shape parameter
        if not self.shape or len(self.shape) != 2:
            raise ValueError("shape must be a tuple of (height, width)")
        # Validate the class names
        if not all(isinstance(cls, str) for cls in self.classes):
            raise ValueError("All class names must be strings")
        # Ensure the dataframe is not empty
        if len(self.df) == 0:
            raise ValueError("DataFrame df cannot be empty")
        # Ensure the dataframe contains the required columns
        if 'sat_image_path' not in self.df.columns or 'mask_path' not in self.df.columns:
            raise ValueError("DataFrame must contain 'sat_image_path' and 'mask_path' columns")


class Dataloader(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        # Determine the indices of the batch
        start = i * self.batch_size
        stop = min((i + 1) * self.batch_size, len(self.indexes))  # Prevent index out of range
        batch_indexes = self.indexes[start:stop]

        # Collect batch data
        data = [self.dataset[j] for j in batch_indexes]

        # Transpose list of lists to create batch
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        # Convert batch to float32
        batch = [tf.cast(item, tf.float32) for item in batch]
        
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)