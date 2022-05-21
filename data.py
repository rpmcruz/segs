from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import os

class KITTISegmentation(Dataset):
    nclasses = 34

    def __init__(self, root, transform=None):
        self.root = root
        self.files = os.listdir(os.path.join(self.root, 'kitti', 'semantics', 'training', 'semantic'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        x = imread(os.path.join(self.root, 'kitti', 'semantics', 'training', 'image_2', filename)).astype(np.float32) / 255
        y = imread(os.path.join(self.root, 'kitti', 'semantics', 'training', 'semantic', filename))
        y = np.stack([y == i for i in range(self.nclasses)], 2)
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = KITTISegmentation('/data')
    x, y = ds[0]
    print('x:', x.dtype, x.shape, x.min(), x.max())
    print('y:', y.dtype, y.shape, y.min(), y.max())
    plt.subplot(5, 6, 1)
    plt.imshow(x)
    for k in range(5*6-1):
        plt.subplot(5, 6, k+2)
        plt.imshow(y[..., k], cmap='gray')
        plt.title(f'class {k}')
    plt.show()
