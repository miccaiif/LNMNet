import numpy as np
import torch.utils.data as data_utils
from torchvision import transforms
import os
import cv2
from PIL import Image

path_mean = [0.6185205578804016, 0.3677789568901062, 0.7136943936347961]
path_std = [0.23521704971790314, 0.2494743913412094, 0.17246422171592712]


class PerSlideBags(data_utils.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.files = os.listdir(self.root)
        self.transform = transform
        self.num_examples = len(self.files)
        self._tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(self.root + self.files[index], cv2.IMREAD_UNCHANGED)
        # print(self.root)
        img = Image.fromarray(img)
        # print(str(index))
        if self.transform is not None:
            img = self.transform(img)
        # img = self._tensor(img)

        return img


class ALLSlideBags(data_utils.Dataset):
    def __init__(self, root, seed=1, train=True, positive='P', bag_length=50):
        self.train = train
        self.root = root
        self.positive = positive
        self.r = np.random.RandomState(seed)
        self.bag_length = bag_length

        if self.train:
            self.all_train_bags_list, self.all_train_labels_list = self._create_allbags()
        else:
            self.all_test_bags_list, self.all_test_labels_list = self._create_allbags()

    def _create_perbags(self, path):
        if self.train:
            dataset = PerSlideBags(root=path,
                                   train=True,
                                   transform=transforms.Compose([
                                       transforms.ColorJitter(brightness=0.5,
                                                              contrast=0.5,
                                                              saturation=0.5,
                                                              hue=0.2),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(40),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=path_mean, std=path_std)]))
            loader = data_utils.DataLoader(dataset=dataset, batch_size=self.bag_length, shuffle=True)
        else:
            dataset = PerSlideBags(root=path,
                                   train=False,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize(mean=path_mean, std=path_std)]))
            loader = data_utils.DataLoader(dataset=dataset, batch_size=self.bag_length, shuffle=True)

        for batch_idx, batch_data in enumerate(loader):
            all_imgs_perbag = batch_data
            if batch_idx == 0:
                break

        per_bags_list = []
        per_bags_list.append(all_imgs_perbag)
        # print(self.root)

        return per_bags_list

    def _create_allbags(self):
        train_bags_list = []
        train_labels_list = []
        for filename_class in os.listdir(self.root):
            for filename_slide in os.listdir(self.root + filename_class + '/'):
                # print('uploading '+filename_slide + ' ' + str(self.bag_length) + ' Per Slide ')
                path_perbags = self.root + filename_class + '/' + filename_slide + '/'
                label_bags = filename_class == self.positive
                train_labels_list.append(label_bags)
                train_bags_list.append(self._create_perbags(path_perbags))

        all_bags_list = [b for a in train_bags_list for b in a]
        all_labels_list = [val for val in train_labels_list]

        return all_bags_list, all_labels_list

    def __len__(self):
        if self.train:
            return len(self.all_train_bags_list)
        else:
            return len(self.all_test_bags_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.all_train_bags_list[index]
            label = self.all_train_labels_list[index]
        else:
            bag = self.all_test_bags_list[index]
            label = self.all_test_labels_list[index]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(ALLSlideBags(
        seed=1,
        root='./train_dataset/',
        bag_length=10,
        train=True),
        batch_size=1,
        shuffle=True)

    test_loader = data_utils.DataLoader(ALLSlideBags(
        seed=1,
        root='./val_dataset/',
        bag_length=10,
        train=False),
        batch_size=1,
        shuffle=True)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label.numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label.numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
