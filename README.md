# Deep-Learning-MNIST---Handwritten-Digit-Recognition

An implementation of multilayer neural network using keras with an Accuracy: mean=98.960 std=0.097, n=5 using 5-crossfold Validation and using the built-in evaluation of 99.13

## About MNIST dataset:
MNIST [(Modified National Institute of Standards and Technology database)](https://medium.com/r/?url=http%3A%2F%2Fyann.lecun.com%2Fexdb%2Fmnist%2F) is probably one of the most popular datasets among machine learning and deep learning enthusiasts. The MNIST dataset contains 60,000 small square 28×28 pixel grayscale training images of handwritten digits from 0 to 9 and 10,000 images for testing. So, the MNIST dataset has 10 different classes.
<br/><br/>
## Steps:

1. Import the libraries and load the dataset: Importing the necessary libraries, packages and MNIST dataset
2. Preprocess the data
3. Create the model
4. Train and Evaluate the Model
5. Saving the model
6. Make Predictions

### Check out the detailed steps at my medium story [Deep Learning Project — Handwritten Digit Recognition using Python](https://medium.com/@aditijain0424/deep-learning-project-handwritten-digit-recognition-using-python-26da7ed11d1c)
<br/><br/>
## Summary of Sequential model

![Scummary](Images/Summary%20of%20the%20Model.png)
<br/><br/>
## Accuracy

**Accuracy** using **5-crossfold Validation** is mean=98.960 std=0.097, n=5 and using the built-in evaluation of **99.13**

![Custom number prediction](https://github.com/Joy2469/Deep-Learning-MNIST---Handwritten-Digit-Recognition/blob/master/Images/accuarcy%20custom.png)

![prediction](Images/accuracy%20with%20custom%20data.png)

<br/><br/>
## Prediction
### A. Dataset images
![Data Set Prediction](https://github.com/Joy2469/Deep-Learning-MNIST---Handwritten-Digit-Recognition/blob/master/Images/data%20set%20image%20prediction.png)

### B. Testing with Custom Number

![Custom number prediction](Images/TestNumber.png)
<br/>
![prediction](Images/prediction.png)
<br/><br/>


# Run
```
python3 predict.py
```


# Resources:
[Deep Learning Introduction](https://medium.com/r/?url=https%3A%2F%2Fwww.forbes.com%2Fsites%2Fbernardmarr%2F2018%2F10%2F01%2Fwhat-is-deep-learning-ai-a-simple-guide-with-8-practical-examples%2F%235a233f778d4b)<br/>
[Install Tensorflow](https://medium.com/@cran2367/install-and-setup-tensorflow-2-0-2c4914b9a265)<br/>
[Why Data Normalizing](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)<br/>
[One-Hot Code](https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fwhy-one-hot-encode-data-in-machine-learning%2F)<br/>
[Understanding of Convolutional Neural Network (CNN)](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148%20https://www.youtube.com/watch?v=YRhxdVk_sIs)<br/>
[CNN layers](https://medium.com/r/?url=https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Fkeras%2Flayers%2FLayer)<br/>
[K-cross Validation](https://medium.com/r/?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DTIgfjmp-4BA)<br/>
[Plotting Graphs](https://medium.com/r/?url=https%3A%2F%2Fmatplotlib.org%2Fapi%2Fpyplot_api.html)<br/>
