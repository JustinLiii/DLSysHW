import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:,::-1]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        x, y = img.shape[0], img.shape[1]
        ret_img = np.zeros_like(img)
        
        if abs(shift_x) > x or abs(shift_y) > y:
            return ret_img
        
        else:
            ret_img[max(0, -shift_x) : min(x-shift_x, x), max(0, -shift_y) : min(y-shift_y, y)] \
                = img[max(0, shift_x) : min(x+shift_x, x), max(0, shift_y) : min(y+shift_y, y)]
            return ret_img
        ### END YOUR SOLUTION
