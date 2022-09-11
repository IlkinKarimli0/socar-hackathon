import numpy as np 
import pickle
import time


#TODO Softmax activation function implementation
##*   <Favourable cost function for that activation
##*   Do them modifiable/>
#TODO Recall scoring system <( on / off )/> 
#TODO F1 scoring system
#TODO it would be better to add running time option to the model
#TODO system for shuffling and seperating data into train, dev and , test sets


class power_estimator_DNN(object):
    #initializer 
    def __init__(self,sizes, activations = None ,   cost_function = 'binary_cross_entropy' ,
                                                    param_init_type = None,
                                                    batch_size = False,

                                                    dropout = False,
                                                    keep_prob=1,

                                                    L2_regularization = False,
                                                    lambd = 0,

                                                    optimizer = 'GD',
                                                    beta1 = 0.8,
                                                    beta2 = 0.9,
                                                    epsilon = 1e-8,

                                                    learning_rate_decay=False,
                                                    decay_rate = 0.3,
                                                    decay_type = "unscheduled",
                                                    lr_decay_time_interval = 100,
                                                    
                                                    recall = False,
                                                    show_running_time_per_iteration = False,
                                                    show_running_time = False
                                                    ):

        
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.caches = dict() 
        self.cost_function = cost_function
        self.batch_size = batch_size
        

        if activations == None:         self.layer_activations = self.default_layer_activations_init(sizes)
        else:                           self.layer_activations = activations
        

        if param_init_type == None:     self.param_init_type = 'default' 
        else:                           self.param_init_type = param_init_type



        if dropout == False :           
                                        self.dropout_forward = self.dropout_empty
                                        self.dropout_backward = self.dropout_empty
                                        self.keep_prob = 1

        else:                           self.keep_prob = keep_prob
            

        if L2_regularization == False:
                                        self.regularization_cost = self.null_regularization
                                        self.regularize_gradients = self.null_regularization
                                        self.lambd = 0

        else :                          self.lambd = lambd
            
        self.weights , self.biases  =   self.parameters_initializer()

        if optimizer =='GD':
            self.add_momentum = self.add_momentum_null

        elif optimizer == 'momentum':
            self.beta1 = beta1
            self.velocities = self.initialize_velocity()

        elif optimizer =='RMSprop': 
            self.beta2 = beta2
            self.epsilon =epsilon
            self.decaying_averages = self.initialize_velocity()
            self.add_momentum = self.add_momentum_null
            self.update_param = self.RMSprop
        
        elif optimizer == 'Adam':
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon =epsilon
            self.velocities = self.initialize_velocity()
            self.decaying_averages = self.initialize_velocity()
            self.update_param = self.RMSprop


        if learning_rate_decay:
            self.decay_rate = decay_rate
            if decay_type == 'unscheduled':
                self.learning_rate_decay = self.continues_learning_rate_decay
            elif decay_type =='scheduled':
                self.lr_decay_time_interval = lr_decay_time_interval
                self.learning_rate_decay = self.scheduled_learning_rate_decay
        else:
            self.learning_rate_decay = self.null_lr_decay

    #This function initalize parameter with needed matrix shapes
    #* it refers to multiple_method function which is give us opportunity to initialize parameters with different methods
    def parameters_initializer(self):
        weights = dict()  ;  biases = dict()
        for l in range(1,self.num_layers):
            print(self.sizes[l],self.sizes[l-1])
            weights['W'+str(l)] = np.random.randn(self.sizes[l],self.sizes[l-1])* self.multiple_method(l)
            biases['b'+str(l)] = np.zeros((self.sizes[l],1))

        return weights,biases
    
    #weight initializer methods
    def multiple_method(self,l):
        if   self.param_init_type == 'default' :   return 0.1
        elif self.param_init_type == 'xavier'  :   return np.sqrt(1/self.sizes[l-1])
        elif self.param_init_type == 'he'      :   return np.sqrt(2/self.sizes[l-1])            

    #this function return the dictionary that contain number of layer and activation function for this layer
    def default_layer_activations_init(self,size):
        return {  num_layer:"relu"  if (num_layer < self.num_layers - 1) else 'sigmoid' for num_layer in range(1,self.num_layers) }

    def initialize_velocity(self):
        velocities = dict()
        for layer_num in range(1,self.num_layers):
            velocities["dW" + str(layer_num)] = np.zeros((self.sizes[layer_num],self.sizes[layer_num-1]),dtype=np.float64)
            velocities["db" + str(layer_num)] = np.zeros((self.sizes[layer_num],1),dtype=np.float64)

        return velocities


    
    def dropout_forward(self,A,layer_num,keep_prob):
        if layer_num != self.num_layers - 1 : #we dont want to add drop out mask to the last layer
            mask = np.random.rand(A.shape[0],A.shape[1])
            mask = (mask <= keep_prob).astype(int)
            A = np.multiply(A,mask)
            A = A / keep_prob
            self.caches['mask'+str(layer_num)] = mask
        return A

    def dropout_empty(self,A,*args):
        return A

    
    #Basically give us the output from network if param A is input
    def feed_forward(self,A,keep_prob=1):

        for layer_num in range(1,self.num_layers):
            Z = np.dot(self.weights['W'+str(layer_num)],A) 
            Z = Z + self.biases['b'+str(layer_num)]
            A = self.activation(Z,self.layer_activations[layer_num])

            A = self.dropout_forward(A,layer_num,keep_prob) 

            ## Add to cache
            self.caches['Z'+str(layer_num)] = Z ; self.caches['A'+str(layer_num)] = A 
        return A #output of network | Y_hat
    
    #These are activation functions for NN
    def activation(self,Z,activation):
        if activation == 'relu':
            return np.maximum(0,Z)
        elif activation == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif activation == 'softmax':
            t = np.exp(Z)
            return ( t/np.sum(t) )


    def regularization_cost(self,m):
        sum_of_weights = 0
        for weight in self.weights.values():
            sum_of_weights += np.sum(np.square(weight))
        L2_regularization_cost = (1/(2*m))*self.lambd*sum_of_weights

        return L2_regularization_cost

    #usable cost functions
    def cost(self,X,Y):
        """param X : Input that will be given to network , Function itself does forward propagation steps and compute cost
           param Y : Wanted output corresponds to given input data. Cost will be computed by This Y and Y_hat which is output of NN for X input"""
        Y_hat = self.feed_forward(X)
        m = Y.shape[1]
        
        if self.cost_function == 'binary_cross_entropy':
            cost = (-1/m)*np.sum( np.multiply(Y,np.log(Y_hat)) + np.multiply( (1-Y) , np.log(1-Y_hat) )) ; cost = np.squeeze(cost)
        elif self.cost_function == 'mse':
            cost = (1/m)*np.sum(np.square(Y-Y_hat)) ; cost = np.squeeze(cost) 
        else:
            raise Exception('No such cost function yet')
        
        cost = cost + self.regularization_cost(m)
        return cost

        
    # function basically return Partial derivative of Cost with respect to last activation

    def cost_derivative(self,last_A,Y):
        """Param last_A : Activation of last layer
           Param Y      : Output"""
        if self.cost_function =='binary_cross_entropy':
            return - (np.divide(Y, last_A) - np.divide(1 - Y, 1 - last_A))
        elif self.cost_function == 'mse':
            return ( last_A - Y )

    def dropout_backward(self,dA_l_prev,layer_num):
        if layer_num-1 != 0:
            mask_l = self.caches['mask'+str(layer_num-1)]
            dA_l_prev = np.multiply(dA_l_prev,mask_l)
            dA_l_prev = dA_l_prev/self.keep_prob

        return dA_l_prev


    # This function propagates NN Backward in order to compute derivatives with chain rule for each parameter 
    def backward_prop(self, dA_l, layer_num):
        """param  dA_l : activation derivative of given layer
           param layer_num : layer number """
        dZ = dA_l* self.activation_derivative(self.caches['Z'+str(layer_num)],self.layer_activations[layer_num])
        m = dA_l.shape[1]
        grad_w_l =(1/m)*np.dot(dZ , self.caches['A'+str(layer_num - 1)].T)
        grad_b_l = (1/m)*np.sum(dZ,axis=1, keepdims=True)
        dA_l_prev =  np.dot(self.weights['W'+str(layer_num)].T,dZ)
        
        #Dropout masking in backward
        dA_l_prev = self.dropout_backward(dA_l_prev,layer_num)
        
        return grad_w_l,grad_b_l,dA_l_prev


    #Derivative of activation function respect to its input Z
    def activation_derivative(self,Z,activation_function):
        if activation_function == 'relu':
            Z[Z<0] = 0; Z[Z>0] = 1 
            return Z
        elif activation_function == 'sigmoid':
            return self.activation(Z,'sigmoid')*(1-self.activation(Z,'sigmoid'))
            #computationally simplified version

    def regularize_gradients(self,m,layer_num):
        reg_grad_w = (self.lambd/m) * self.weights['W'+str(layer_num)]# Regularizing gradients of weights that coming from backrop
        return reg_grad_w

    def null_regularization(self,*args):
        return 0

    #update the parameters with computed gradients 
    def update_param(self,grad_w,grad_b,layer_num,lr):
        #TODO L2 reg ll change 
        """param  grad_w , grad_b : gradients of parameters
           param layer_num        : layer number 
           param lr               : learning rate """


        # TODO Idea: Make backprop algo reversible so with calculated prev activation derivative make backprop algo goes back,forward and back again 
        layer_num = str(layer_num) # we need string type of layer number so as to concatnate string for calling key of parameters dictionary 
        self.weights['W'+layer_num] = self.weights['W'+layer_num] - lr*grad_w
        self.biases['b'+layer_num] = self.biases['b'+layer_num] - lr*grad_b

    def add_momentum(self,grad_w,grad_b,layer_num,t=2):
        layer_num = str(layer_num)
        self.velocities["dW" + layer_num] = self.beta1 * self.velocities["dW" + layer_num] + (1-self.beta1) * grad_w
        self.velocities["db" + layer_num] = self.beta1 * self.velocities["db" + layer_num] + (1-self.beta1) * grad_b

        #print('Here is velocity ',self.velocities['dW'+layer_num] , '\n Here is denom ',(1-self.beta1**self.epoch_num))
        #print('beta1 {} , epoch num {} '.format(self.beta1,self.epoch_num))
        #corrected versions
        corrected_w_velocity = self.velocities["dW" + layer_num]/(1-self.beta1**self.epoch_num)
        corrected_b_velocity = self.velocities["db" + layer_num]/(1-self.beta1**self.epoch_num)

        return corrected_w_velocity,corrected_b_velocity
    
    def add_momentum_null(self,grad_w,grad_b,*args):
        return grad_w,grad_b

    def RMSprop(self,grad_w,grad_b,layer_num,lr,t=2):
        #Root Mean Squared Propagation 
        layer_num = str(layer_num)

        self.decaying_averages["dW" + layer_num] = self.beta2 * self.decaying_averages["dW" + layer_num] + (1-self.beta2) * grad_w**2
        self.decaying_averages["db" + layer_num] = self.beta2 * self.decaying_averages["db" + layer_num] + (1-self.beta2)*  grad_b**2
        #corrected versions
        corrected_avg_w= self.decaying_averages["dW" + layer_num]/(1-self.beta2**self.epoch_num)
        corrected_avg_b= self.decaying_averages["db" + layer_num]/(1-self.beta2**self.epoch_num)

        #print('decaying average of b ',self.decaying_averages["db" + layer_num] )
        self.weights['W'+layer_num] = self.weights['W'+layer_num] - lr*grad_w/(np.sqrt(corrected_avg_w)+self.epsilon)
        self.biases ['b'+layer_num] = self.biases ['b'+layer_num] - lr*grad_b/(np.sqrt(corrected_avg_b)+self.epsilon)

    def continues_learning_rate_decay(self,lr,iter):
        return 1/(1 + self.decay_rate**iter) * lr
    
    def scheduled_learning_rate_decay(self,lr,iter):
        return 1/(1+self.decay_rate*np.floor(iter/self.lr_decay_time_interval))*lr

    def null_lr_decay(self,lr,*args):
        return lr

    def fit(self,X,Y,decay_rate = 0.3,lr = 0.0001,epochs=1000 , X_test = None , Y_test = None ,mini_iter =1):
        m = X.shape[1]#number of batches
        assert (m == Y.shape[1]) , "Unmatched In out batch size"
        
        #checking for Batch gradient decent
        if self.batch_size == False:    self.batch_size=m
        
        num_mini_batches = int(np.floor( m / self.batch_size )) #number of mini batches that fit to batch

        for iter in range(1,epochs+1):
            self.epoch_num = iter # we gonna use this variable inside root mean squared propagation ( Part of adam optimizer used for correction of decaying averages and velocities)
            lr = 1/(1 + decay_rate**iter) * lr
            lr = self.learning_rate_decay(lr,iter)
            #shuffle the data
            permutation = list(np.random.permutation(m))
            X = X[:, permutation]
            Y = Y[:, permutation].reshape((1, m))

            for mb_num in range(num_mini_batches): #Splitting batch to mini batches and train on them
                mini_batch_X = X[:,   mb_num * self.batch_size : ( mb_num+1 ) *self.batch_size]
                mini_batch_Y = Y[:,   mb_num * self.batch_size : ( mb_num+1 ) *self.batch_size]

                self.train(mini_batch_X,mini_batch_Y,lr=lr,iterations=mini_iter)

            if m%self.batch_size !=0:
                #train the last part of batch
                mini_batch_X = X[:,   m - m%self.batch_size :]
                mini_batch_Y = Y[:,   m - m%self.batch_size :]
                self.train(mini_batch_X,mini_batch_Y,lr=lr,iterations=mini_iter)

            if iter% (epochs/10) ==0:
                print('\nEpoch: {}     Cost: {}'.format(iter,self.cost(X,Y)),end=' ')    
                self.evaluate(X,Y)
        if X_test is not None:
            self.evaluate(X_test,Y_test)

        #Saving parameters dictionary to file 
        weights = open("weights.pkl", "wb")     ;   biases = open("biases.pkl", "wb")   
        pickle.dump(self.weights, weights)      ;   pickle.dump(self.biases, biases) 
        weights.close()                         ;   biases.close() 
    
    def recall(self,Y_hat,Y):
        pass
    
    def train(self,X,Y,lr,iterations=1):
        
        self.caches['A0'] = X
        for iter in range(iterations):
            A_l = self.feed_forward(X,self.keep_prob)
            dA_l = self.cost_derivative(A_l,Y)
            for layer_num in reversed(range(1,self.num_layers)):
                grad_w,grad_b,dA_l = self.backward_prop(dA_l,layer_num)
                grad_w += self.regularize_gradients(X.shape[1],layer_num)
                grad_w,grad_b = self.add_momentum(grad_w,grad_b,layer_num)
                
                self.update_param(grad_w,grad_b,layer_num, lr = lr)
            

    def decide(self, A_last):
        """This function will decide final prediction based on output and last activation function"""
        last_l_num = list(self.layer_activations)[-1]#layer number of last layer

        if self.layer_activations[last_l_num] == 'sigmoid':
            A_last[A_last<=0.5] = 0 ; A_last[A_last>0.5] = 1
        elif self.layer_activations[last_l_num]  == 'relu':
            pass
        return A_last

    #If test datasets are given, during training we will score NN on train set and once training is completed 
    #It measures score of NN on test set and gives final accuracy
    def evaluate(self,X,Y):
        Y_hat = self.feed_forward(X)
        Y_hat = self.decide(Y_hat)
        count =  0
        for i in range(Y.shape[1]):
            if Y_hat[0][i] ==  Y[0][i]  :
                count+=1
        (Y_hat == Y)
            
        print( '\n  {} / {} Accuracy : {}% '.format(count,Y.shape[1],count/Y.shape[1]*100))

    def predict(self,X):
        #Reading parameters from file to file 
        weights = open("weights.pkl", "rb")  ;  biases = open("weights.pkl", "rb")
        self.weights= pickle.load(weights)   ;  self.biases= pickle.load(biases)
        #computing output
        output = self.feed_forward(X)
        return output


class TestNetwork(power_estimator_DNN):

    def __init__(self, sizes, lambd, activations=None, cost_function='binary_cross_entropy', param_init_type=None, batch_size=32, keep_prop=1 , optimizer = None,beta=0,beta1=0.9,beta2=0.999):
        
        super().__init__(sizes, activations, cost_function, param_init_type, batch_size, keep_prop)
        self.lambd = lambd
        if    optimizer == None:             self.optimizer = 'Gradient_descent'# Parameters will be updated normally, 
                                                                                # We can still choose its type SGD, mini batch or
                                                                                # batch gradient descent with batch size
        else:                                self.optimizer = optimizer
        

    def initialize_velocity(self):
        if self.optimizer == 'momentum':
            self.velocity = dict()
            for l in range(1,self.num_layers):
                self.velocity["dW" + str(l)] = np.zeros((self.weights['W' + str(l)].shape[0],self.weights['W' + str(l)].shape[1]))
                self.velocity["db" + str(l)] =np.zeros((self.biases['b' + str(l)].shape[0],self.biases['b' + str(l)].shape[1]))

        pass

    def parameters_initializer(self):
        return super().parameters_initializer()

    def multiple_method(self, l):
        return super().multiple_method(l)

    def default_layer_activations_init(self, size):
        return super().default_layer_activations_init(size)
    
    def feed_forward(self, A, keep_prob=1):
        return super().feed_forward(A, keep_prob)

    def activation(self, Z, activation):
        return super().activation(Z, activation)

    def cost(self, X, Y):
        m = Y.shape[1]
        normal_cost = super().cost(X, Y)
        sum_of_weights = 0
        for weight in self.weights.values():
            sum_of_weights += np.sum(np.square(weight))
        L2_regularization_cost = (1/(2*m))*self.lambd*sum_of_weights
        return normal_cost+L2_regularization_cost

    def cost_derivative(self, last_A, Y):
        return super().cost_derivative(last_A, Y)
    
    def backward_prop(self, dA_l, layer_num):
        return super().backward_prop(dA_l, layer_num)
        

    def activation_derivative(self, Z, activation_function):
        return super().activation_derivative(Z, activation_function)

    def update_param(self, grad_w, grad_b, layer_num, lr):
        return super().update_param(grad_w, grad_b, layer_num, lr)
    
    def fit(self, X, Y, lr=0.0001, epochs=1000, X_test=None, Y_test=None):
        return super().fit(X, Y, lr, epochs, X_test, Y_test)

    def train(self, X, Y, lr, iterations=1):
        self.caches['A0'] = X
        m = X.shape[1]
        for iter in range(iterations):
            A_l = self.feed_forward(X,self.keep_prob)
            dA_l = self.cost_derivative(A_l,Y)
            for layer_num in reversed(range(1,self.num_layers)):
                grad_w,grad_b,dA_l = self.backward_prop(dA_l,layer_num)
                
                self.update_param(grad_w,grad_b,layer_num, lr = lr)

    def decide(self, A_last):
        return super().decide(A_last)

    def evaluate(self, X, Y):
        return super().evaluate(X, Y)

    def predict(self, X):
        return super().predict(X)
    

    
    
        
        
            



        




    


            






    