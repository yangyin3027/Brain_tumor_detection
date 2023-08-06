import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets
from torchvision.utils import make_grid

def transform(train=True):
    if train:
        augs = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((128,128)),
                        transforms.Grayscale(num_output_channels=3), # transform (h, w) image to (3, h, w) compartible to pretrained cnn
                        transforms.ColorJitter(brightness=0.6),
                        transforms.GaussianBlur((3,3),sigma=(0.1,1)),
                        transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(degrees=0,translate=(0.05, 0.1),
                                            shear=0.1),
                        transforms.ToTensor(),

                ])
        return augs
    
    else:
        val_augs = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((128,128)),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),   
                    ])
        return val_augs

class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)   
        return img, label

def get_dataloader(dataset, batch_size=32):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def plot_augmented_imgs(img, transform):

    fig = plt.figure( constrained_layout=True)
    subfigs = fig.subfigures(2,1)

    aug_imgs = [transform(img) for _ in range(16)]
    grid = make_grid(aug_imgs, nrows=6, normalize=False)

    imgs = [img, grid.permute(1,2,0)]
    titles = ['Original Image', 'Augmented Images']

    for i in range(2):
        subfigs[i].add_subplot()
        plt.imshow(imgs[i], 'gray')
        plt.axis('off')
        subfigs[i].suptitle(titles[i])
    plt.show()
        
def compute_datasets_stats(root, batch_size=32,
                           resize=(64,64)):
    '''
    root folder with structured datasets
    '''

    augs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize)
    ])

    img_dataset = datasets.ImageFolder(root=root,
                                       transform=augs)
    img_loader = torch.utils.data.DataLoader(img_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    
    psum = torch.zeros(3, dtype=torch.float32)
    psum_sq = torch.zeros(3, dtype=torch.float32)

    for data, _ in img_loader:
        psum += data.sum(dim=[0,2,3])
        psum_sq += (data**2).sum(dim=[0,2,3])
    
    count = len(img_loader) * batch_size * resize[0] * resize[1]

    mean = psum / count
    var = psum_sq / count - mean**2
    std = torch.sqrt(var)
    
    mean, std = mean.detach().numpy(), std.detach().numpy()
    
    return mean, std