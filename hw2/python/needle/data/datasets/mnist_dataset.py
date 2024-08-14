from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        import gzip
        import struct
    
        with gzip.open(image_filename, 'rb') as f:
            _, size, nrows, ncols = struct.unpack('>iiii',f.read(4*4)) # type: ignore
            feature_size = nrows * ncols
            X = np.frombuffer(f.read(), dtype=np.uint8) # type: ignore
            X = X.astype(np.float32)
            X = X / 255.0
            X = X.reshape(size, nrows, ncols, 1) # different from previous hw, (B, H, W,)
                    
        with gzip.open(label_filename, 'rb') as f:
            _, size = struct.unpack('>ii',f.read(4*2)) # type: ignore
            y = np.frombuffer(f.read(), dtype=np.uint8) # type: ignore
                
        self.X = X
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return (self.apply_transforms(self.X[index]), self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION