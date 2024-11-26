# Facial Recognition Using Python and TensorFlow

Video overview: <https://www.youtube.com/watch?v=lab8V430_io>

In this project I created a convolutional neural network to analyze and classify images of people's faces in the lfw dataset.
The dataset initially contained over 5000 classes and most of these classes did not have sufficient amount of images in them for me to work on a deep neural network setting.
I filtered out the classes that have less than 50 images in them and ended up with 12 classes with a total of 1560 images.
The data was highly unbalanced so i used the class weights to mitigate the effects.
My model was able to achieve mostly around 80 percent accuracy and precision even though the data was relatively small for a deep neural network and highly unbalanced.

## Requirements
tensorflow
scikit-learn
opencv

## Acknowledgements
Dataset I used: <https://www.kaggle.com/datasets/atulanandjha/lfwpeople>