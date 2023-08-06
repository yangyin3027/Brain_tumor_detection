import os
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Images:
    '''
    Create an Images object with method to load_images, plot_examples, and crop_images 
    '''
    def __init__(self, root):
        '''
        root: folder contained yes and no folders
        '''
        self.root = root
    
    def datasets(self):
        imgs = self.refine_imgs()[0]
        labels = self.refine_imgs()[1]
        return imgs, labels

    def read_images(self):
        imgs = []
        labels = []
        
        for sub in os.listdir(self.root):
            if sub in ['yes', 'no']:
                res = self.read_images_sub(sub)
                imgs.extend(res[0])
                labels.extend(res[1])
            else:
                continue
        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels
    
    def read_images_sub(self, sub):
        imgs = []
        labels = []
        
        img_path = os.path.join(self.root, sub)
        
        for f in os.listdir(img_path):
            if not f.startswith('.'):
                img_file = os.path.join(img_path, f)
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (128, 128))
                
                imgs.append(img)
                labels.append(sub)
                
        return imgs, labels
    
    def refine_imgs(self, threshold=30):
        '''
            Find the boundary box to crop black boarder of the image.
            cv2.findContour used to find the largest contour, and use it to find the boundary box

            threshold: hyperparameter that will determine the final accuracy of cropping
        '''
        imgs, labels = self.read_images()
        
        new_imgs = []
        
        for i in range(len(imgs)):
            img = imgs[i]
            gray = cv2.GaussianBlur(img, (5,5), 0)
            res, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # findContours works best on binary images
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            new_img = img[y:y+h, x:x+w]
            new_img = cv2.resize(new_img, (128,128))
            new_imgs.append(new_img)
        
        return np.stack(new_imgs), labels
    
    def plot_examples(self, ncols=8):
        imgs, labels = self.datasets()
        
        yes_imgs = imgs[labels=='yes'][:ncols]
        no_imgs = imgs[labels=='no'][:ncols]
        
        fig = plt.figure(
                         layout='constrained')
        subfigs = fig.subfigures(2,1, 
                                 hspace=0.001)
        
        imgs_group = [yes_imgs, no_imgs]
        titles = ['Tumor: Yes', 'Tumor: No']
        for subfig, imgs, title in zip(subfigs, imgs_group, titles):
            subfig.add_subplot()
            grid = make_grid_numpy(imgs, ncols=ncols)
            subfig.suptitle(title)
            plt.imshow(grid, 'gray')
            plt.axis('off')
        plt.show()   

def make_grid_numpy(imgs, ncols=8, padding=6, padding_values=0):
            
    if isinstance(imgs, list):
        imgs = np.stack(imgs)

    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, -1)

    h, w, c = imgs.shape[1:]   

    nums = len(imgs)
    nrows = math.ceil(nums/ncols)

    grid = np.full((h*nrows+(nrows+1)*padding, w*ncols+(ncols+1)*padding, c), padding_values)

    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if k >= nums:
                break
            row = padding*(i+1) + h*i
            col = padding*(j+1) + w*j
            grid[row:row+h, col:col+w, :] = imgs[k, ...]
            k += 1
    return grid.squeeze() 

def stratify_train_val_split(imgs, labels, train_size=0.8, shuffle=True):
    
    for data in [imgs, labels]: 
        if isinstance(data, (list, np.ndarray)):
            if isinstance(imgs, list):
                data = np.concatenate(data)
        else:
            raise ValueError('imgs or labels have to be list or np.ndarray type')
        
    
    pos_imgs = imgs[labels==1]
    pos_labels = labels[labels==1]
    
    neg_imgs = imgs[labels==0]
    neg_labels = labels[labels==0]
    
    def permutate(imgs, labels):
        perm = np.random.permutation(len(imgs))
        return imgs[perm], labels[perm]
                    
    if shuffle:
        pos_imgs, pos_labels = permutate(pos_imgs, pos_labels)
        neg_imgs, neg_labels = permutate(neg_imgs, neg_labels)
    
    num_train = int(min(len(pos_imgs), len(neg_imgs)) * train_size)
    
    X_train_pos, X_valid_pos = pos_imgs[:num_train], pos_imgs[num_train:]
    y_train_pos, y_valid_pos = pos_labels[:num_train], pos_labels[num_train:]
    
    X_train_neg, X_valid_neg = neg_imgs[:num_train], neg_imgs[num_train:]
    y_train_neg, y_valid_neg = neg_labels[:num_train], neg_labels[num_train:]
    
    X_train = np.concatenate((X_train_pos, X_train_neg))
    y_train = np.concatenate((y_train_pos, y_train_neg))
    X_valid = np.concatenate((X_valid_pos, X_valid_neg))
    y_valid = np.concatenate((y_valid_pos, y_valid_neg))
    
    X_train, y_train = permutate(X_train, y_train)
    X_valid, y_valid = permutate(X_valid, y_valid)
    
    return X_train, X_valid, y_train, y_valid

if __name__ == '__main__':
    datasets = Images('./images')
    imgs, labels = datasets.datasets()

    datasets.plot_examples()