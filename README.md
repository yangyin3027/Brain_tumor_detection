# brain_tumor_detection
Dataset was downloaded from [kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). 
## Preprocess the images
- The images have varied black border. The findContours function in OpenCV-Python was used to find the largest contour and thus define the bounding box of each image. Using bounding box images were cropped to remove black border. 
## Build models
- This is an very small dataset. Even with image data augmentation and transfer learning, it is challenged to reach high accuracy.
- Instead, pretrained neural networks, such as VGG16, was used to extract bottleneck features of the images. The deep learning extracted features were then feed to traditional machine learning models for classification.
- Here, Random Forest Classifier using the extracted bottlenect features reached the best accurcy score 
