
'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
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

'''

#Importing necessary modules
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#Defining the neural network class 
class NN: 
     
    parameters = list() #Member of class - a list that stores the parameters i.e weights and biases
    
    #init_params() initializes parameters based on the dimensions given by neuron_count_per_layer
    # neuron_count_per_layer is list that stores the number of neurons in each layer
    def init_params(self,neuron_count_per_layer):
        #Set a random seed for consistent results
        np.random.seed(42)
        num_layers = len(neuron_count_per_layer)
        params = [None for i in range(2*(num_layers-1))]
        for i in range(1,num_layers):
            #Initialize weights of ith layer
            params[2*i-2]=np.random.randn(neuron_count_per_layer[i], neuron_count_per_layer[i-1]) * 0.15
            #Initialize biases of ith layer
            params[2*i-1]=np.zeros((neuron_count_per_layer[i], 1))
        return params
    
  
    # relu() returns the rectified linear unit value for the given input Z i.e max(0,z) 
    #ReLU is the activation function for every layer except the last one
    #The input Z is also returned for caching (required for backprop)
    def relu(self,Z):
        return np.maximum(0,Z),Z
    
    # sigmoid() returns the sigmoid value for the given input
    #ReLU is the activation function for every layer except the last one
    #The input Z is also returned for caching (required for backprop)
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z)),Z
    
    #compute_activation() returns the output/activation for a layer given its weights and biases
    #A tuple containing the inputs A,weight,bias and Z is returned for later use in back propogation
    
    """Inputs:
    A : o/p of prev. layer
    weight,bias : parameters for the layer
    activation:activation fn. to be used"""
    def compute_activation(self,A,weight,bias,activation):
        #Linear computation
        Z = weight@A + bias
        cache1 = (A,weight,bias)
        #Activation part
        if activation=='relu':
            A1,cache2 = self.relu(Z)
        else:
            A1,cache2 = self.sigmoid(Z)
        return A1,(cache1,cache2)
    
    #compute_gradients() calculates the gradients of weights and biases for a given layer making use of cached values
    """Inputs:
    dA :backprop o/p of next layer
    vals : Values cached in forward prop
    activation:activation fn. used in forward prop"""
    def compute_gradients(self,dA,vals,activation):
        cache1,cache2 = vals
        #backprop differentiation for ReLU activation function
        if activation=='relu':
            dZ = np.array(dA,copy=True)
            dZ[cache2<=0]=0
        #backprop differentiation for Sigmoid activation function
        if activation=='sigmoid':
            sig = 1/(1+np.exp(-cache2))
            dZ = dA * sig * (1-sig)
        #Calculations of different derivatives 
        A_prev, W, b = cache1
        x = A_prev.shape[1]
        dW = 1 / x * dZ @ A_prev.T
        db = 1 / x * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ
        return dA_prev,dW,db

            
    # forward_propogation() is the wrapper fn. for forward propogation. It returns the output of the network i.e o/p of last layer and also a cache of different values needed for backprop 
    """
    Inputs : 
    X : The training set
    parameters : Weights and biases of different layers"""
    def forward_propogation(self,X,parameters):
        num_layers = len(parameters)//2
        back_prop_values = []
        A = X
        #Compute output through all the ReLU layers
        for i in range(1,num_layers):
            prev_act = A
            weight = parameters[2*i-2]
            bias = parameters[2*i-1]
            A,back_prop_value = self.compute_activation(prev_act,weight,bias,activation='relu')
            back_prop_values.append(back_prop_value)
        
    
        #Compute output through the final layer ie Sigmoid layer
        A,back_prop_value = self.compute_activation(A,parameters[-2],parameters[-1],activation='sigmoid')
        back_prop_values.append(back_prop_value) 
        return A,back_prop_values
    
    # back_propogation() is the wrapper fn. for backward propogation.It computes the necessary derivatives and updates the parameters based on the learning rate.The updated parameters are returned
    """Y:True labels
    activations:predicted labels
    parameters : weights and biases of diff layers
    back_prop_values:values cached in forward prop
    alpha : learning rate
    v,s:Adam parameters"""
    def back_propogation(self,Y,activations,parameters,back_prop_values,alpha,v,s):
        
        #Set adam params
        beta1 = 0.9; 
        beta2 = 0.999;
        t=2;
        
        e = 1e-5; #Used for mathematical stability in divison
        
        
        #Computing the necessary derivatives
        gradients = {} #Dictionary for storing the gradients for each parameter
        num_layers = len(parameters)//2
        Y.reshape(activations.shape)
        
        #Differentiate the activations i.e cost function 
        dA = - (np.divide(Y, activations) - np.divide(1 - Y, 1 - activations))
        
        #Compute gradients for last layer i.e Sigmoid layer
        vals = back_prop_values[num_layers-1]
        gradients["dA" + str(num_layers-1)], gradients["dW" + str(num_layers)], gradients["db" + str(num_layers)] = self.compute_gradients(dA,vals,'sigmoid')
        
        #Compute gradients for the ReLU layers N-1 to 1
        for layer in reversed(range(num_layers-1)):
            vals = back_prop_values[layer]
            gradients["dA" + str(layer)], gradients["dW" + str(layer + 1)], gradients["db" + str(layer + 1)] = self.compute_gradients(gradients['dA'+str(layer+1)], vals, 'relu')
        
        
        b_corrected_1 = [None for i in range(2*(num_layers-1)+2)]
        b_corrected_2 = [None for i in range(2*(num_layers-1)+2)]
        
        #Updating the parameters with the computed gradients - using Adam optimization technique
        for i in range(1,num_layers+1): 
            #Update v
            v[2*i-2] = beta1 * v[2*i-2] + (1 - beta1) * gradients['dW' + str(i)]
            v[2*i-1] = beta1 * v[2*i-1] + (1 - beta1) * gradients['db' + str(i)]

            #Bias-corrected first estimate
            b_corrected_1[2*i-2] = v[2*i-2] / (1 - beta1**t)
            b_corrected_1[2*i-1] = v[2*i-1] / (1 - beta1**t)
        

            #Update s
            s[2*i-2] = beta2 * s[2*i-2] + (1 - beta2) * gradients['dW' + str(i)]**2
            s[2*i-1] = beta2 * s[2*i-1] + (1 - beta2) * gradients['db' + str(i)]**2
            

            #Bias-corrected second estimate
            b_corrected_2[2*i-2] = s[2*i-2] / (1 - beta2**t)
            b_corrected_2[2*i-1] = s[2*i-1] / (1 - beta2**t)

            #Update parameters
            parameters[2*i-2] = parameters[2*i-2] - alpha * b_corrected_1[2*i-2] / (np.sqrt(b_corrected_2[2*i-2]) + e)
            parameters[2*i-1] = parameters[2*i-1] - alpha * b_corrected_1[2*i-1] / (np.sqrt(b_corrected_2[2*i-1]) + e)

        return parameters
    
    #init_adam_params() initializes the parameters necessary for Adam Optimzer
    # neuron_count_per_layer is list that stores the number of neurons in each layer
    def init_adam_params(self,neuron_count_per_layer):    
        num_layers = len(neuron_count_per_layer)
        v = [None for i in range(2*(num_layers-1))]
        s = [None for i in range(2*(num_layers-1))]
        for i in range(1,num_layers):
            v[2*i-2] = np.zeros_like(self.parameters[2*i-2])
            v[2*i-1] = np.zeros_like(self.parameters[2*i-1])
            s[2*i-2] = np.zeros_like(self.parameters[2*i-2])
            s[2*i-1] = np.zeros_like(self.parameters[2*i-1])
        return v,s


    ''' X and Y are dataframes '''

    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
       
        #Number of epochs
        num_itertations = 750

        #Learning rate
        alpha = 0.02

        #Network architecture
        neuron_count_per_layer = [9,30,30,25,1]
    
        #Init NN parameters
        self.parameters = self.init_params(neuron_count_per_layer)
        #Init
        v,s = self.init_adam_params(neuron_count_per_layer)
        
        #Making necessary changes to dimensions
        X = np.transpose(np.array(X))
        Y = np.array(Y)
        Y = np.reshape(Y,(1,Y.shape[0]))
        
        for i in range(1,num_itertations+1):

            #Forward prop
            activations,back_prop_values = self.forward_propogation(X,self.parameters)
            #Backward prop
            self.parameters = self.back_propogation(Y,activations,self.parameters,back_prop_values,alpha,v,s)
            cost = self.calc_cost(activations,Y)
            #Stop the training if cost is very low - to prevent any underflow in subsequent iterations
            if cost<3:
                break
    
    #Computes the cross entropy loss function
    def calc_cost(self,A,Y):
        return np.squeeze(-1 / len(Y) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)))
    
    def predict(self,X):

        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values 

        yhat is a list of the predicted value for df X
        """
        yhat = []
        X = np.transpose(np.array(X))
        yhat = self.forward_propogation(X,self.parameters)[0][0] 
        return yhat

    def CM(y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.5):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        print("Confusion Matrix : ")
        print(cm)
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
   



def main():
    #Read CSV file
    df = pd.read_csv('PP_LBW_Dataset.csv',index_col=0)
    #Extract input and output features
    X = df.iloc[:,:-1]
    y = df.iloc[:, -1]
    #Generate train-test split (70%-30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

    #Create a neural network object
    nn = NN()

    #Train the network
    nn.fit(X_train,y_train)

    #Compute training accuracy
    print('Training results')
    y_hat = nn.predict(X_train)
    NN.CM(y_train.tolist(),y_hat)

    #Compute testing accuracy
    print('\n\nTesting results')
    y_hat = nn.predict(X_test)
    NN.CM(y_test.tolist(),y_hat) 
    

#Call the main function
main()




