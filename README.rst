=======
Linear Region Attack
=======

The Linear Region attack is a powerful white-box adversarial attack that
exploits knowledge about the geometry of neural networks to find minimal
adversarial perturbations without doing gradient descent.

This repository provides an efficient GPU impelementation of the Linear Region
attack. If you find our attack useful or use this code, please cite
**"Scaling up the randomized gradient free adversarial attack reveals
overestimation of robustness using established attacks"**.

BibTeX
------

.. code-block::

  @article{todo,
  }

Requirements
------------

This impelementation requires Python 3.6 or newer, NumPy and JAX.
Before installing JAX, you need to install jaxlib with GPU support:

.. code-block:: bash

  PYTHON_VERSION=cp36
  CUDA_VERSION=cuda100
  PLATFORM=linux_x86_64
  BASE_URL='https://storage.googleapis.com/jax-wheels'
  python3 -m pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.11-$PYTHON_VERSION-none-$PLATFORM.whl

  python3 -m pip install --upgrade jax

For details regarding the installation of JAX, please check the `JAX readme <https://github.com/google/jax#installation>`_.

We have successfully used ``Python 3.6``, ``NumPy 1.16``, ``JAX 0.1.21`` and ``jaxlib 0.1.11``.

Usage
-----

To run the attack on a 10-layer convnet trained on CIFAR10 for the first image in the CIFAR-10 test set, just run this:

.. code-block:: bash

  ./main.py cifar_convnet --regions 40  # just for illustration; we recommend more regions, e.g. 400

Note: To run the example, you need CIFAR-10:

.. code-block:: bash

  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -zxvf cifar-10-python.tar.gz
