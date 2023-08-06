from utils import *
from image_transform import *

imgs, labels = Images('./images').datasets()

transform = transform(train=True)

plot_augmented_imgs(imgs[np.random.randint(len(imgs))],
                    transform=transform)