Neural Network designed by PESU-MI_0273_1930_2012

Table of contents:
- Introduction 
- Pre-Processing
- Building up the network
- Results

Introduction:

As a part of our 3rd assignment, we have built an ANN which helps in classifying a given patient as a case of Low Birth Weight(LBW) or not. 

Preprocessing:

The given dataset had many nulls and as a result of which, we cleaned the dataset by replacing the missing values with the mean, mode or median of the column. For columns like weight, HP and BP, we used the mean value whereas, for columns like Community,Delivery phase,IFA and Education we used the mode value and for age we used the median.

Building up the network:

As an initial step of building up the network, we divided the dataset into train and test set(70-30). 
The hyperparameters are set as follows:
1. ANN layers:
	An input layer with 9 neurons(9 features)
	3 hidden layers with 30,30 and 25 neurons in the increasing order of layer number
	An output layer with one neuron
2. The learning rate: 0.02
3. Number of epochs: 750
4.Weight Initialization : Random * 0.15
5.Bias Initialization : Zeroes
6.The activation function used in all the layers except the last one is ReLU and in the last layer, sigmoid has been used.
7.Cost function : Cross entropy
8.Adam optimization has been implemented.
Its parameters:
	beta1 : 0.9
	beta2 : 0.999
	t = 2


The implementation follows the standard method.

The "fit" function initializes the hyperparameters and the parameters(weights and bias) through "init_params" function. 
Following this, for each iteration, forward propagation happens by calling the "forward_propagation" function. During forward propagation, the summation of product of weights and inputs and its summation with the bias is computed along with the activation value, in the "compute_activation" function. Finally the function returns the back propagation values and the activation function values. 
Now the above values along with the learning rate,alpha is passed to the back bropagation function function wherein the gradients are calulated and the weights are updated and finally, the function returns the updated parameters. 

In order to predict the values for the test set, we call the "predict" function and the forward propagation function is called and output values(yhat) is returned.
In order to check the accuracy, we call the "CM" function by passing the output values and this printds the confusion matrix, precision, Recall and F1 score.
The code is well documneted with detailed comments at appropriate places.


After a considerable amount of tweeking of the model, we were able to achieve an F1 score of 0.951 in training and 0.884 in testing.

The implementation stands out with respect to its simplicity of design and an appreciable accuracy with the given limited dataset. Adam optimzation has been implemented. As a result, the training is much quicker and more effective. It is also an aspect beyond the basic/standard implementations.

Steps to execute:

1.Run preprocessing_code.py on a Linux terminal with the command python3 preprocessing_code.py
This results in creation of a preprocessed dataset CSV file in the directory.
2.Run main.py with python3 main.py
The programs outputs the results of testing and training.

Step 1 can be skipped as the preprocessed file is already placed in src directory in case the user wants to run main.py directly.