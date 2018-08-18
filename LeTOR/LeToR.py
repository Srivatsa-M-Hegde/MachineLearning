import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans



#Reading data from csv files
df_input=pd.read_csv("./Querylevelnorm_X.csv", sep=',',header=None)
df_output=pd.read_csv("./Querylevelnorm_t.csv", sep=',',header=None)
df_output.columns = ['y']
df = pd.concat([df_input, df_output] , axis =1 )
train, validate, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
#Creating train sets
train_input =  train[train.columns[0:46]]
train_output = train['y']
trainsize, _ = train_input.shape
#Creating validation sets
validate_input =  validate[validate.columns[0:46]]
validate_output = validate['y']
validatesize , _ = validate_input.shape
#Creating test sets
test_input =  test[test.columns[0:46]]
test_output = test['y']
testsize , _ = test_input.shape



##Initializing input and output train data
input_data_train = np.array(train_input)
##output_data = np.array(train_output)
output_data_train = train_output.reshape([-1,1])
##Initializing input and output validation data
input_data_validate = np.array(validate_input)
##output_data = np.array(train_output)
output_data_validate = validate_output.reshape([-1,1])
##Initializing input and output test data
input_data_test = np.array(test_input)
##output_data = np.array(train_output)
output_data_test = test_output.reshape([-1,1])



##setting input and output data as training input
input_data = input_data_train
output_data = output_data_train

##Function to compute the design matrix
def compute_design_matrix(X, centers, spreads):
 # use broadcast 
 basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers),axis=2) / (-2)).T
 # insert ones to the 1st col
 return np.insert(basis_func_outputs, 0, 1, axis=1)


K=6
################################################################################################3
##Function to return the 'K' centres from the K means clustering
##Returns an [M,1,D] array 
def calculate_centers(K):
   kmeans = KMeans(n_clusters=K)
   KOut = kmeans.fit(train_input)
   KOut.cluster_centers_
   return np.array(KOut.cluster_centers_)  

##Function to return the Diagonal matrix with diagonals as the variance * k  
##Returns an [M,D,D] array
def calculate_spreads(K):
    diagonal = np.zeros(shape = (46,46))
    k1=5 
    for i in range(0,46) : 
        val = np.var(input_data[:,i])
        diagonal[i,i]= val * k1                
    lst2 = []
    for i in range(K):
      lst2.append(diagonal)
          
    return lst2

##Function to calculate validation error
def calculate_errors_validate(w , design_matrix_validate , validate_output ) :
    error = (1/2)*(np.sum(np.power((np.matmul(design_matrix_validate  , w) - validate_output) , 2)))   
    error = np.power(error *(2/validatesize) , 0.5)
    return error

##Function to calculate test error 
def calculate_errors_test(w , design_matrix_test  , test_output) :
    error = (1/2)*(np.sum(np.power((np.matmul(design_matrix_test , w) - test_output)  ,2)))
    error = np.power(error *(2/testsize) , 0.5)
    return error

##Function to calculate the closed form solution 
def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix), np.matmul(design_matrix.T, output_data)).flatten()

##Function to calculate the weights using Stochastic Gradient Decent   
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda,design_matrix,output_data , K):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = 1 is equivalent to standard gradient descent
    # Using minibatch_size = N is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.random.randn(1, (K+1)) * 0.1
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary
    lst = []
    for epoch in range(num_epochs):
      #Early stop implementation ( note - This method makes sense only for large epoch values . But we choose only small epoch values of 10000)
      #if(epoch % 10 == 0) :
             #error_old = early_stop(20 , error_old)
             #if(early_stop(20 , error_old) ==1 ) :
                    # Stop the SGD Algorithm if the number of times validation error increases is equal to the patience value
      for i in range(int(N / minibatch_size)):
         lower_bound = i * minibatch_size
         upper_bound = min((i+1)*minibatch_size, N)
         Phi = design_matrix[lower_bound : upper_bound, :]
         t = output_data[lower_bound : upper_bound , :]
         E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi )
         E = (E_D + L2_lambda * weights) / minibatch_size
         weights = weights - learning_rate * E
      
      lst.append(np.linalg.norm(E))            
    
    return weights.flatten() , lst  
###########################################################################################
N , D = input_data.shape 

##Calling the functions to calculate centers and spreads 
X = input_data[np.newaxis, :, :]
centers = calculate_centers(K) 
centers = centers[:, np.newaxis, :]
spreads = calculate_spreads(K)


## Hyper Parameter tuning 
############################################################################################

# Learning Rate -
## Choosing too small a value for learning rate may increase the time for convergence and the likelihood of being stuck
## at the local minima , whereas choosing too large a value may cause the error to become NaN
Learning_Rate = [0.0001, 0.001, 0.01, 0.05 , 0.1 , 0.5 ]

# K Value -
## Choosing the K value decides how the clustering happens . Its essential to choose the right K value for an optimal 
## clustering . The centers and the spread are affected by the 'K' value we choose for clustering , hence can be optimized
## by a simple grid search for the best value
K_Vector = [  3 , 4 , 5 , 7 , 10 ] 

# Regularization Term
## Choosing the 'lambda' value allows us to ensure we do not overfit the data by reducing the variance but with a tradeoff
## for variance 
lambda_vector = [ 0.001 , 0.01 , 0.1 , 1 , 2 ] 

# Early Stopping 
## This is done by analyzing the rate at which the validation error changes . When the validation error stops decreasing , 
## we stop the training process (using SGD) . This helps minimize the time for training even further .
p=10
def early_stop (p , count , error_old , error_new):
        stop = 0 
        if ( error_old < error_new ) :
            count = count + 1 
        if (count >= p) :
            stop = 1 
        return stop # stop =1 will stop the training process

##Grid search for parameter tuning 
ValidationError_Vector = []
for i in range(len(K_Vector)) :
    for j in range(len(lambda_vector)) :
        for k in range(len(Learning_Rate)) :
            
            X = input_data[np.newaxis, :, :]
            centers = calculate_centers(K_Vector[i]) 
            centers = centers[:, np.newaxis, :]
            spreads = calculate_spreads(K_Vector[i])
            #Computing the Design matrix for train set using the variable parameter 'K'
            design_matrix_train = compute_design_matrix(X, centers, spreads)
            Weight_Train, Y = SGD_sol(learning_rate=Learning_Rate[k],minibatch_size=256,num_epochs=50,L2_lambda=lambda_vector[j],design_matrix=design_matrix_train,output_data=output_data_train, K=K_Vector[i])
            #Calculating the design matrix for validation set
            design_matrix_validate = compute_design_matrix(input_data_validate[np.newaxis, : , :] , centers , spreads )
            #Calculating the error on the validation set
            error = calculate_errors_validate(Weight_Train.T , design_matrix_validate , output_data_validate) 
            #Storing all the validation set errors in a list 
            ValidationError_Vector.append([error,i,j,k] )     

            
##Finding the best selection of parameters 
minimum = ValidationError_Vector[0][0]
for i in range(len(ValidationError_Vector)) :
             minimum = min(minimum, ValidationError_Vector[i][0])
for i in range(len(ValidationError_Vector)) :
             if (ValidationError_Vector[i][0] == minimum) :
                     index = [ValidationError_Vector[i][1],ValidationError_Vector[i][2] , ValidationError_Vector[i][3]]
#K_final is the best value of K
K_final=K_Vector[index[0]]
#lambda_final is the best value of K
lambda_final=lambda_vector[index[1]]
#Learning_Rate_final is the best value of K
Learning_Rate_final=Learning_Rate[index[2]]
print("The best hyperparameters are : K = " , K_Vector[index[0]] , "lambda= " ,  lambda_vector[index[1]] , " Learning Rate = " , Learning_Rate[index[2]])
        
    
########################################################
##Test Error Calculation

#Calculating the design matrix for test set
X = input_data[np.newaxis, :, :]
centers = calculate_centers(K_final) 
centers = centers[:, np.newaxis, :]
spreads = calculate_spreads(K_final)
#Computing the Design matrix for train set using the variable parameter 'K'
design_matrix_train = compute_design_matrix(X, centers, spreads)
design_matrix_test = compute_design_matrix(input_data_test[np.newaxis, : , :] , centers , spreads )
#Calculating the weights for the best selection of hyper parameters that returned the least RMS error on the validation set
Weight_Train, Y = SGD_sol(learning_rate=Learning_Rate_final ,minibatch_size=256,num_epochs=1000,L2_lambda=lambda_final,design_matrix=design_matrix_train,output_data=output_data_train, K=K_final )
print("Closed form solution of the weights" ,closed_form_sol(L2_lambda=lambda_final,design_matrix=design_matrix_train,output_data=output_data_train))
print("Gradient decent solution of the weights" , Weight_Train)
#Calculating the test error
print("The root mean square test error is  " ,  calculate_errors_test(Weight_Train , design_matrix_test  , test_output) )
#%matplotlib inline
plt.plot(range(len(Y)),Y)   
