from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs1, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs1 = selected_attrs1
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []

            for attr_name in self.selected_attrs1:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')


            if i < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB') 
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(image_dir, attr_path, selected_attrs1, image_size=256, 
               batch_size=16, mode='train', num_workers=1, shuffle=False):
    """Build and return a data loader."""

    transform = []
    transform.append(T.Resize([image_size,image_size]))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, selected_attrs1, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    return data_loader