from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool
from sklearn.utils.class_weight import compute_class_weight
from skimage.io import imread
from functools import partial
from itertools import repeat
import tensorflow as tf
import numpy as np
import os
import random

def multioutput_encode(y_labels,num_classes):
    reg_mat=np.tril(np.ones([num_classes,num_classes-1]),-1)
    y_encoded=reg_mat[y_labels]
    return np.hsplit(y_encoded[:,0],num_classes-1)

def multioutput_encode_doel3(y_labels,num_classes):
    reg_mat=np.ones([num_classes,num_classes])-np.triu(np.ones([num_classes,num_classes]),+1)+np.tril(np.ones([num_classes,num_classes]),-1)
    y_encoded=reg_mat[y_labels]
    split=np.hsplit(y_encoded[:,0],num_classes)
    return split

def sample_weights(y):
    weights=[]
    for i in y:
        w=compute_class_weight('balanced',np.unique(i),i.ravel())
        j=np.array(i,dtype=int)
        if np.unique(i).size==2 and np.unique(i)[1]==2:
            aux=w[j-1]
        else:
            aux=w[j]
        weights.append(aux)
    return weights

def categorical_ensemble(y):
    encode_mat=np.array([[0.8,0.2,0],[0.1,0.8,0.1],[0,0.2,0.8]])
    j=np.array(y,dtype=int)
    cat_split=[np.squeeze(encode_mat[i]) for i in j]
    return cat_split

def SORDencoder(y_labels,class_labels,num_classes, metric='absolute'):
    """
    Function that encode the class label as soft ordinal regression encode. 
    """
    if metric=='squared':
        cost_matrix=np.power(np.tile(class_labels, (num_classes, 1)) - np.reshape(class_labels, (-1,1)),2) / num_classes
    if metric=='squared_log':
        #Through the common use of a 0 label, I add an really low value to avoid infinite logarithm
        cost_matrix=np.power(np.abs(np.log(np.tile(class_labels+0.00000001, (num_classes, 1))) - np.log(np.reshape(class_labels, (-1,1))+0.00000001)),2) / num_classes
    else:
        #By default, absolute metric is used
        cost_matrix=np.abs(np.tile(class_labels, (num_classes, 1)) - np.reshape(class_labels, (-1,1))) / num_classes

    #Calculates the softmax of each value for each row
    encode_mat=np.exp(-cost_matrix)/np.reshape(np.sum( np.exp(-cost_matrix),axis=1),(-1,1))
    y_encoded=encode_mat[y_labels]
    return y_encoded[:]

def generate_random_augmentation(p, shape):
    """
    Generates the data of a random augmentation for ImageDataGenerator.
    """
    aug = {}

    if 'rotation_range' in p:
        aug['theta'] = random.uniform(-p['rotation_range'], p['rotation_range'])

    if 'width_shift_range' in p:
        aug['ty'] = random.uniform(-p['width_shift_range'] * shape[1], p['width_shift_range'] * shape[1])

    if 'height_shift_range' in p:
        aug['tx'] = random.uniform(-p['height_shift_range'] * shape[0], p['height_shift_range'] * shape[0])

    if 'shear_range' in p:
        aug['shear'] = random.uniform(-p['shear_range'], p['shear_range'])

    if 'zoom_range' in p:
        aug['zy'] = aug['zx'] = random.uniform(1 - p['zoom_range'], 1 + p['zoom_range'])

    if 'flip_horizontal' in p:
        aug['flip_horizontal'] = p['flip_horizontagenerate_random_augmentationl']

    if 'flip_vertical' in p:
        aug['flip_vertical'] = p['flip_vertical']

    if 'channel_shift_range' in p:
        aug['channel_shift_intencity'] = random.uniform(-p['channel_shift_range'], p['channel_shift_range'])

    if 'brightness_range' in p:
        aug['brightness'] = random.uniform(-p['brightness'], p['brightness'])

    return aug


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

# Process a single image

def process_data(augmentation, x):
    # Apply data augmentation
    if len(augmentation) > 0:
        if 'crop' in augmentation:
            x = random_crop(x, (augmentation['crop'], augmentation['crop']))

        x = ImageDataGenerator().apply_transform(x, generate_random_augmentation(augmentation, shape=x.shape))

    return x

class SmallGenerator(Sequence):
    """
    Class to define a generator for small image size datasets
    """
    def __init__(self, x, y, num_classes, mean=None, std=None, batch_size=128, augmentation={}, workers=7, encode='one_hot', labels=[],soft_ordinal_config='absolute', ensemble=False, ensemble_train=False, ensemble_type='regression'):
        self._x = x
        self._ensemble = ensemble
        self._ensemble_train = ensemble_train
        self._ensemble_type= ensemble_type
        self._y = y
        self._num_classes = num_classes
        if self._ensemble:
            if self._ensemble_type=='doel3':
                self._y = multioutput_encode_doel3(self._y, self._num_classes)          
            else:
                self._y = multioutput_encode(self._y, self._num_classes)
        if self._ensemble_train:
            self._sample_weights = sample_weights(self._y)
        self._mean = mean
        self._std = std
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._workers = workers
        self._p = Pool(self._workers)
        self._encode = encode
        self._labels = labels
        self._soft_ordinal_config = soft_ordinal_config
        super(SmallGenerator, self).__init__()

    def __len__(self):
        return int(np.ceil(len(self._x) / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]

        if not self._ensemble:
            batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        else:
            batch_y = [i[idx * self._batch_size:(idx + 1) * self._batch_size] for i in self._y]
            if self._ensemble_type=='doel3':
                batch_y = categorical_ensemble(batch_y)

        func = partial(process_data, self._augmentation)
        batch_x = np.array(self._p.map(func, batch_x))
                    
        # Standardize
        if self._mean and self._std:
            batch_x = (batch_x - self._mean) / self._std

        if self._ensemble_train:
            batch_weight = [i[idx * self._batch_size:(idx + 1) * self._batch_size] for i in self._sample_weights]
            return np.array(batch_x), batch_y, batch_weight

        if self._ensemble:
             return np.array(batch_x), batch_y

        elif self._encode=='one_hot':
            batch_y = to_categorical(batch_y, num_classes=self._num_classes)
        
        elif self._encode=='soft_ordinal':
            batch_y = SORDencoder(batch_y,self._labels,self._num_classes, metric=self._soft_ordinal_config)


        return np.array(batch_x), np.array(batch_y)

    def __del__(self):
        if self._p is not None:
            self._p.close()
            self._p.terminate()
            self._p.join()


def process_data_path(augmentation, force_rgb, base_path, path):
    img = imread(os.path.join(base_path, path))

    # Convert to RGB if grayscale
    if force_rgb and len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)

    # Apply data augmentation
    if len(augmentation) > 0:
        img = ImageDataGenerator().apply_transform(img, generate_random_augmentation(augmentation, shape=img.shape))

    return img

class BigGenerator(Sequence):
    """
    Class to define a generator for big image size datasets
    """
    def __init__(self, df, base_path, num_classes, x_col='x', y_col='y', mean=None, std=None, batch_size=128, augmentation={}, workers=7, encode='one_hot', force_rgb=True, labels=[],soft_ordinal_config='absolute', ensemble=False, ensemble_train=False,ensemble_type='regression'):
        self._df = df
        self._base_path = base_path
        self._num_classes = num_classes
        self._x_col = x_col
        self._y_col = y_col
        self._ensemble = ensemble
        self._ensemble_train=ensemble_train
        self._ensemble_type=ensemble_type
        if self._ensemble_train:
            if self._ensemble_type=='doel3':
                self._sample_weights = sample_weights( multioutput_encode_doel3(np.transpose([np.array(self._df[self._y_col])]), self._num_classes))
            else:
                self._sample_weights = sample_weights( multioutput_encode(np.transpose([np.array(self._df[self._y_col])]), self._num_classes))
                        
        self._mean = mean
        self._std = std
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._workers = workers
        self._p = Pool(self._workers)
        self._encode = encode
        self._force_rgb = force_rgb
        self._labels=labels
        self._soft_ordinal_config=soft_ordinal_config
        super(BigGenerator, self).__init__()


    def __len__(self):
        return int(np.ceil(self._df.shape[0] / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_paths = self._df.iloc[idx * self._batch_size : (idx + 1) * self._batch_size][self._x_col]
        batch_y = self._df.iloc[idx * self._batch_size : (idx + 1) * self._batch_size][self._y_col]

        if self._ensemble:
            if self._ensemble_type=='doel3':
                batch_y = multioutput_encode_doel3(np.transpose([np.array(batch_y)]), self._num_classes)
                batch_y=categorical_ensemble(batch_y)
            else:
                batch_y = multioutput_encode(np.transpose([np.array(batch_y)]), self._num_classes)

        # Load batch images using multiprocessing
        func = partial(process_data_path, self._augmentation, self._force_rgb, self._base_path)
        batch_x = np.array(self._p.map(func, batch_paths))

        # Standardize
        if self._mean and self._std:
            batch_x = (batch_x - self._mean) / self._std

        if self._ensemble_train:
            batch_weight = [i[idx * self._batch_size:(idx + 1) * self._batch_size] for i in self._sample_weights]
            return np.array(batch_x), batch_y,batch_weight

        if self._ensemble:
             return np.array(batch_x), batch_y

        if self._encode=='one_hot':
            batch_y = to_categorical(batch_y, num_classes=self._num_classes)
        
        elif self._encode=='soft_ordinal':
            batch_y = SORDencoder(batch_y,self._labels,self._num_classes,metric=self._soft_ordinal_config)
        
        return np.array(batch_x), np.array(batch_y)

    def __del__(self):
        if self._p is not None:
            self._p.close()
            self._p.terminate()
            self._p.join()
