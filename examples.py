import pickle
import sys
import os
import numpy as onp
import logging
import jax
from functools import partial

from staxmod import Conv, Dense, Flatten, Relu
from staxmod import serial

from utils import is_device_array


train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
]

test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
]


def load_cifar10(*, train):
    if train:
        filenames = train_list
    else:
        filenames = test_list

    images = []
    labels = []

    try:
        for filename, checksum in filenames:
            path = os.path.join('cifar-10-batches-py', filename)
            path = os.path.expanduser(path)
            with open(path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
            images.append(entry['data'])
            labels.extend(entry['labels'])
    except FileNotFoundError:
        logging.error('Could not load CIFAR. Run the following commands to download it:\n'
                      '\n'
                      'wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n'
                      'tar -zxvf cifar-10-python.tar.gz')
        sys.exit(1)

    images = onp.vstack(images).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    labels = onp.asarray(labels, dtype=onp.int32)

    assert images.dtype == onp.uint8
    images = images.astype(onp.float32) / 255.
    return images, labels


def ConvNet():
    return serial(
        Conv(96, (3, 3), padding='SAME'), Relu,
        Conv(96, (3, 3), padding='SAME'), Relu,
        Conv(192, (3, 3), padding='SAME', strides=(2, 2)), Relu,
        Conv(192, (3, 3), padding='SAME'), Relu,
        Conv(192, (3, 3), padding='SAME'), Relu,
        Conv(192, (3, 3), padding='SAME', strides=(2, 2)), Relu,
        Conv(192, (3, 3), padding='SAME'), Relu,
        Conv(384, (2, 2), padding='SAME', strides=(2, 2)), Relu,
        Flatten,
        Dense(1200), Relu,
        Dense(10))


def load_params(path):
    with open(path, 'rb') as f:
        params = pickle.load(f)
    params = jax.tree_map(jax.device_put, params)
    return params


def find_starting_point(images, labels, args, x, label, logits, predict_class):
    strategy = args.nth_likely_class_starting_point
    if strategy is None:
        return find_starting_point_simple_strategy(images, labels, x, label, predict_class)
    return find_starting_point_likely_class_strategy(images, labels, x, label, logits, predict_class, nth=strategy)


def find_starting_point_simple_strategy(images, labels, x, label, predict_class):
    """returns the image in images that is closest to x that has a
    different label and predicted class than the provided label of x"""

    assert x.shape[0] == 1

    assert not is_device_array(x)
    assert not is_device_array(label)

    assert not is_device_array(images)
    assert not is_device_array(labels)
    assert not is_device_array(x)
    assert not is_device_array(label)

    # filter those with the same label
    images = images[labels != label]

    # get closest images from other classes
    diff = images - x
    diff = diff.reshape((diff.shape[0], -1))
    diff = onp.square(diff).sum(axis=-1)
    diff = onp.argsort(diff)
    assert diff.ndim == 1

    for j, index in enumerate(diff):
        logging.info(f'trying {j + 1}. candidate ({index})')
        candidate = images[index][onp.newaxis]
        class_ = jax.device_get(predict_class(candidate).squeeze(axis=0))
        logging.info(f'label = {label}, candidate class = {class_}')
        if class_ != label:
            return candidate, class_


def find_starting_point_likely_class_strategy(images, labels, x, label, logits, predict_class, *, nth):
    assert x.shape[0] == 1

    assert not is_device_array(x)
    assert not is_device_array(label)

    assert not is_device_array(images)
    assert not is_device_array(labels)
    assert not is_device_array(x)
    assert not is_device_array(label)

    # determine nth likely class
    logits = logits.squeeze(axis=0)
    ordered_classes = onp.argsort(logits)
    assert ordered_classes[-1] == label

    assert 2 <= nth <= len(logits)
    nth_class = ordered_classes[-nth]

    # select those from the nth most likely  class
    images = images[labels == nth_class]

    # get closest images from other classes
    diff = images - x
    diff = diff.reshape((diff.shape[0], -1))
    diff = onp.square(diff).sum(axis=-1)
    diff = onp.argsort(diff)
    assert diff.ndim == 1

    for j, index in enumerate(diff):
        logging.info(f'trying {j + 1}. candidate ({index})')
        candidate = images[index][onp.newaxis]
        class_ = jax.device_get(predict_class(candidate).squeeze(axis=0))
        logging.info(f'label = {label}, candidate class = {class_}')
        if class_ != label:
            return candidate, class_


def _cifar_example(architecture, weights=None):
    init, predict = architecture()
    output_shape, params = init((-1, 32, 32, 3))
    if weights is not None:
        params = load_params(weights)
    n_classes = output_shape[-1]
    images, labels = load_cifar10(train=False)
    train_images, train_labels = load_cifar10(train=True)
    assert not is_device_array(train_images) and not is_device_array(train_labels)
    find_starting_point_2 = partial(find_starting_point, train_images, train_labels)
    return n_classes, predict, params, images, labels, find_starting_point_2


def get_cifar_example(load_weights=True):
    weights = 'weights/convnet.pickle' if load_weights else None
    return _cifar_example(ConvNet, weights)


def get_example(name):
    return {
        'cifar_convnet': lambda: get_cifar_example(load_weights=True),
    }[name]()
