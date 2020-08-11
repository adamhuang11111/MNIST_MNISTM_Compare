import pickle as pkl
import numpy as np
from scipy import ndimage
from tensorflow.examples.tutorials.mnist import input_data

def edge_detect_helper(image:np.ndarray, mode):
    sx = ndimage.sobel(image, axis=0, mode=mode)
    sy = ndimage.sobel(image, axis=1, mode=mode)
    sob = np.hypot(sx, sy)
    return sob

def edge_detect(images):
    images_edged = edge_detect_helper(images[:,:,0] / 255, mode='constant')
    return images_edged / images_edged.max()

def _normalize(ndarray, channelwise_means):
    return (ndarray - channelwise_means) / 255

def load_data_org(mnist_path, mnistm_path, pytorch=True, normalize=True, onehot_labels=False):
    """
    Load training and testing data for DANN.
    
    param: mnist_path: path to the folder containing mnist dataset stored in a tensorflow format
    param: mnistm_path: path to the pickled mnistm dataset
    param: normalize: whether to normalize training and testing data using channel-wise means
    """

    mnist = input_data.read_data_sets(mnist_path, one_hot=True)

    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    mnist_train_labels = mnist.train.labels
    mnist_test_labels = mnist.test.labels
    
    # Load MNIST-M
    mnistm = pkl.load(open(mnistm_path, 'rb'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    # mnistm_valid = mnistm['valid']

    # Compute pixel mean for normalizing data
    channelwise_means = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))
    
    # Create a mixed dataset for TSNE visualization
    num_test = 500
    combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
    combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
    combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
            np.tile([0., 1.], [num_test, 1])])
    
    if normalize:
        mnist_train = _normalize(mnist_train, channelwise_means)
        mnist_test = _normalize(mnist_test, channelwise_means)
        mnistm_train = _normalize(mnistm_train, channelwise_means)
        mnistm_test = _normalize(mnistm_test, channelwise_means)
        combined_test_imgs = _normalize(combined_test_imgs, channelwise_means)
        
    if pytorch:
        mnist_train = np.rollaxis(mnist_train, 3, 1)
        mnist_test = np.rollaxis(mnist_test, 3, 1)
        mnistm_train = np.rollaxis(mnistm_train, 3, 1)
        mnistm_test = np.rollaxis(mnistm_test, 3, 1)
        combined_test_imgs = np.rollaxis(combined_test_imgs, 3, 1)
        
    if not onehot_labels:
        mnist_train_labels = mnist_train_labels.argmax(axis=1)
        mnist_test_labels = mnist_test_labels.argmax(axis=1)
        combined_test_labels = combined_test_labels.argmax(axis=1)
        combined_test_domain = combined_test_domain.argmax(axis=1)
    
    return (mnist_train, mnist_train_labels), (mnist_test, mnist_test_labels), (mnistm_train, mnistm_test), (combined_test_imgs, combined_test_labels, combined_test_domain)
