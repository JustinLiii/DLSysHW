"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    import gzip
    
    with gzip.open(image_filesname, 'rb') as f:
        _, size, nrows, ncols = struct.unpack('>iiii',f.read(4*4)) # type: ignore
        feature_size = nrows * ncols
        X = np.frombuffer(f.read(), dtype=np.uint8) # type: ignore
        X = X.astype(np.float32)
        X = X / 255.0
        X = X.reshape(size, feature_size)
                
    with gzip.open(label_filename, 'rb') as f:
        _, size = struct.unpack('>ii',f.read(4*2)) # type: ignore
        y = np.frombuffer(f.read(), dtype=np.uint8) # type: ignore
            
    return (X,y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    l = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,))) - ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(l) / l.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X: np.ndarray, y: np.ndarray, W1: ndl.Tensor, W2: ndl.Tensor, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    b = 0
    while b < X.shape[0]:
        if b + batch <= X.shape[0]:
            xx = X[b:b+batch] # (batch_size, input_dim)
            yy = y[b:b+batch] # (batch_size,)
        else:
            xx = X[b:] # (batch_size, input_dim)
            yy = y[b:] # (batch_size,)
            
        # W1 (input_dim, hidden_dim)
        # W2 (hidden_dim, output_dim)

        z = ndl.relu(ndl.Tensor(xx) @ W1) @ W2 # (batch_size, output_dim)
        y_one_hot = np.zeros(z.shape) # 因为z是ndl.Tensor，不能用ones_like
        print(y_one_hot.shape)
        y_one_hot[np.arange(y_one_hot.shape[0]), yy] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        loss = softmax_loss(z, y_one_hot)
        loss.backward()
        W1.data -= lr * W1.grad.data
        W2.data -= lr * W2.grad.data
        
        b += batch
        
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
