import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from matplotlib import pyplot as plt


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    faces_addr = []
    for filename in os.listdir(directory_name):
        faces_addr.append(os.path.join(directory_name, filename))
    return faces_addr


class ORL(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.images, self.labels = self._load_images(self.root)
        self.test_idx = []
        for i in range(40):
            self.test_idx.append(i * 10 + 5)
            self.test_idx.append(i * 10 + 8)
        if split == 'train':
            self.images = [self.images[i] for i in range(400) if i not in self.test_idx]
            self.labels = [self.labels[i] for i in range(400) if i not in self.test_idx]
        elif split == 'test':
            self.images = [self.images[i] for i in self.test_idx]
            self.labels = [self.labels[i] for i in self.test_idx]
        else:
            raise ValueError('split must be "train" or "test"')
        self.transform = transform

    def _load_images(self, root):
        faces = []
        # data/ORL/s1
        for i in range(1, 41):
            faces_addr = read_directory(os.path.join(root, 'ORL', 's' + str(i)))
            for addr in faces_addr:
                faces.append(addr)

        # 读取图片数据,生成列表标签
        images = []
        labels = []
        for index, face in enumerate(faces):
            # enumerate函数可以同时获得索引和值
            image = Image.open(face)
            images.append(image)
            labels.append(int(index / 10 + 1))

        return images, labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # 读取所有人脸文件夹,保存图像地址在faces列表中
    pass

    train_set = ORL(root='data')
    test_set = ORL(root='data', split='test')
    print(len(train_set))
    print(train_set[0][0].size)
    print(train_set[0][1])
    #
    print(f'{np.random.choice(400, 80)}')
