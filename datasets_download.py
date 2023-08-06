import os
import random


def auto_download():    
    download = 'kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection'
    os.system(download)

    unzip = 'unzip ./brain-mri-images-for-brain-tumor-detection.zip -d ./images'
    os.system(unzip)

if __name__ =='__main__':
    os.system('zsh dataset_download.sh')



