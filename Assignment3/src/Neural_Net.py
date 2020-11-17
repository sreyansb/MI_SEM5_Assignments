'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
class NN: 
    parameters = list()
    
    def init_params(self,neuron_count_per_layer):
        num_layers = len(neuron_count_per_layer)
        params = [None for i in range(2*(num_layers-1))]
        for i in range(1,num_layers):
            params[2*i-2]=np.random.randn(neuron_count_per_layer[i], neuron_count_per_layer[i-1]) * 0.01
            params[2*i-1]=np.zeros((neuron_count_per_layer[i], 1))
        return params
    
    #Clean the data by replacing the null values with the mean of the column
    def data_clean(self,df):
        df.columns = df.columns.str.strip()
        for column in df.columns:
            if column in ['Weight','HB','BP']:
                df[column].fillna(value=df[column].mean(), inplace=True)
            elif column in ['Community','Delivery phase','IFA','Education']:
                df[column].fillna(value=df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(value=df[column].median(), inplace=True)
        return df
    
    def relu(self,Z):
        return np.maximum(0,Z),Z
    
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z)),Z
    
    def compute_activation(self,A,weight,bias,activation):
        Z = weight@A + bias
        cache1 = (A,weight,bias)
        if activation=='relu':
            A1,cache2 = self.relu(Z)
        else:
            A1,cache2 = self.sigmoid(Z)
        return A1,(cache1,cache2)
     
    def compute_gradients(self,dA,vals,activation):
        cache1,cache2 = vals
        if activation=='relu':
            dZ = np.array(dA,copy=True)
            #print(cache2.shape,dA.shape)
            dZ[cache2<=0]=0
        if activation=='sigmoid':
            sig = 1/(1+np.exp(-cache2))
            dZ = dA * sig * (1-sig)
        A_prev, W, b = cache1
        x = A_prev.shape[1]
        dW = 1 / x * dZ @ A_prev.T
        db = 1 / x * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W.T @ dZ
        return dA_prev,dW,db
            
    
    def forward_propogation(self,X,parameters):
        num_layers = len(parameters)//2
        back_prop_values = []
        A = X
        
        for i in range(1,num_layers):
            prev_act = A
            weight = parameters[2*i-2]
            bias = parameters[2*i-1]
            A,back_prop_value = self.compute_activation(prev_act,weight,bias,activation='relu')
            back_prop_values.append(back_prop_value)
        
    
        #For Last layer i.e sigmoid 
        A,back_prop_value = self.compute_activation(A,parameters[-2],parameters[-1],activation='sigmoid')
        back_prop_values.append(back_prop_value) 
        #print(len(back_prop_values))
        return A,back_prop_values
    
    def back_propogation(self,Y,activations,parameters,back_prop_values,alpha):
        #Computing the necessary derivatives
        gradients = {}
        num_layers = len(parameters)//2
        Y.reshape(activations.shape)
        dA = - (np.divide(Y, activations) - np.divide(1 - Y, 1 - activations))
        #print(dA.shape)
        vals = back_prop_values[num_layers-1]
        gradients["dA" + str(num_layers-1)], gradients["dW" + str(num_layers)], gradients["db" + str(num_layers)] = self.compute_gradients(dA,vals,'sigmoid')
        for layer in reversed(range(num_layers-1)):
            vals = back_prop_values[layer]
            gradients["dA" + str(layer)], gradients["dW" + str(layer + 1)], gradients["db" + str(layer + 1)] = self.compute_gradients(gradients['dA'+str(layer+1)], vals, 'relu')
        
        #Updating the parameters
        for i in range(1,num_layers+1):
            parameters[2*i-2]=parameters[2*i-2] - alpha * gradients['dW'+str(i)]
            parameters[2*i-1]=parameters[2*i-1] - alpha * gradients['db'+str(i)]
        
        return parameters
        

    def calc_cost(self,A,Y):
        return np.squeeze(-1 / len(Y) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)))

    ''' X and Y are dataframes '''

    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''

        #Clean the data
        X = self.data_clean(X)
        #Set hyperparameters
        num_itertations = 10000
        alpha = 0.05
        
    
        #Init parameters
        neuron_count_per_layer = [9,25,25,25,1]
        self.parameters = self.init_params(neuron_count_per_layer)
        
        #Making necessary changes to dimensions
        X = np.transpose(np.array(X))
        Y = np.array(Y)
        Y = np.reshape(Y,(1,Y.shape[0]))
        
        for i in range(1,num_itertations+1):
            #Fp
            activations,back_prop_values = self.forward_propogation(X,self.parameters)
            #Bp
            self.parameters = self.back_propogation(Y,activations,self.parameters,back_prop_values,alpha)
            #Print Cost after every 500 iters
            if i%500==0:
                print('Cost after iter '+str(i)+ ':' + str(self.calc_cost(activations,Y)/100))
 
        
        
        
    
    def predict(self,X):

        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values 

        yhat is a list of the predicted value for df X
        """
        yhat = []
        X = self.data_clean(X)
        X = np.transpose(np.array(X))
        prob,_ = self.forward_propogation(X,self.parameters)
        for ans in prob[0]:
            if ans>0.5:
                yhat.append(1)
            else:
                yhat.append(0)
        return yhat

    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
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
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")

   




