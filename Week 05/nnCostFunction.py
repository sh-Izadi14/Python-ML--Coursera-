import numpy as np
from sigmoid import sigmoid

def nnCostFunction(nn_params,\
                        input_layer_size, \
                        hidden_layer_size, \
                        num_labels, \
                        X, y, Lambda):
#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#   X, y, Lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network

# Reshape nn_params back into the parameters Theta1 and Theta2
      
      Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)

      Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

      # Setup some useful variables
      m = X.shape[0]
            
      # You need to return the following variables correctly 
      J = 0

      # ====================== YOUR CODE HERE ======================
      # Instructions: You should complete the code by working through the
      #               following parts.
      #
      # Part 1: Feedforward the neural network and return the cost in the
      #         variable J. After implementing Part 1, you can verify that your
      #         cost function computation is correct by verifying the cost
      #         computed in ex4.m
      #
      # Part 2: Implement the backpropagation algorithm to compute the gradients
      #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
      #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
      #         Theta2_grad, respectively. After implementing Part 2, you can check
      #         that your implementation is correct by running checkNNGradients
      #
      #         Note: The vector y passed into the function is a vector of labels
      #               containing values from 1..K. You need to map this vector into a 
      #               binary vector of 1's and 0's to be used with the neural network
      #               cost function.
      #
      #         Hint: We recommend implementing backpropagation using a for-loop
      #               over the training examples if you are implementing it for the 
      #               first time.
      #
      # Part 3: Implement regularization with the cost function and gradients.
      #
      #         Hint: You can implement this around the code for
      #               backpropagation. That is, you can compute the gradients for
      #               the regularization separately and then add them to Theta1_grad
      #               and Theta2_grad from Part 2.
      #

      # Add ones to the X data matrix
      temp = np.ones((m,1))
      a1 = np.c_[temp, X]

      a2 = sigmoid(a1 @ Theta1.T)
      a2 = np.c_[temp, a2]

      a3 = sigmoid(a2 @ Theta2.T)

      for i in range(m):
            y_vec = np.zeros((num_labels,1))
            index = y[i]
            y_vec[index-1,0] = 1 
             
            J = J + (np.log(a3[i,:]).T @ y_vec + np.log(1-a3[i,:]).T @ (1 - y_vec))

      Theta1_reg = Theta1
      Theta1_reg = np.delete(Theta1,0,1)
      
      Theta2_reg = Theta2
      Theta2_reg = np.delete(Theta2,0,1)

      reg_term = (Lambda/(2*m)) * ( sum(sum(Theta1_reg ** 2)) + sum(sum(Theta2_reg ** 2)) )

      J = -(1/m) * J + reg_term

      return J

def nnGradFunction(nn_params, input_layer_size, hidden_layer_size, \
                        num_labels, X, y, Lambda):

      Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)

      Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
      
      # You need to return the following variables correctly 

      Theta1_grad = np.zeros_like(Theta1)
      Theta2_grad = np.zeros_like(Theta2)
      
      # Feedforward Propagation
      m = X.shape[0]
      temp = np.ones((m,1))
      a1 = np.c_[temp, X]

      a2 = sigmoid(a1 @ Theta1.T)
      a2 = np.c_[temp, a2]

      a3 = sigmoid(a2 @ Theta2.T)

      Delta1 = np.zeros_like(Theta1)
      Delta2 = np.zeros_like(Theta2)
      
      for i in range(m):
            y_vec = np.zeros((num_labels,1))
            index = y[i]
            y_vec[index-1,0] = 1 
            
            delta3 = a3[i,:].reshape(y_vec.shape) - y_vec
            
            delta2 = (Theta2.T @ delta3) * (a2[i,:]* (1 - a2[i,:])).reshape((a2.shape[1],1))
            delta2 = delta2[1:]
            
            Delta2 = Delta2 + (delta3 @ a2[i,:].reshape((1,a2.shape[1])))
            
            Delta1 = Delta1 + (delta2 @ a1[i,:].reshape((1,a1.shape[1])))

      Theta1_reg = Theta1
      Theta1_reg = np.delete(Theta1_reg,0,1)
      
      Theta2_reg = Theta2
      Theta2_reg = np.delete(Theta2_reg,0,1)

      if Lambda == 0:
            D2 = (1/m) * Delta2
            Theta2_grad = D2

            D1 = (1/m) * Delta1
            Theta1_grad = D1
      else:
            D20 = (1/m) * Delta2[:,0] # for j = 0
            D2  = (1/m) * Delta2[:,1:] + (Lambda/m)*Theta2_reg # for j >= 1
            Theta2_grad = np.c_[D20, D2]
            
            D10 = (1/m) * Delta1[:,0] # for j = 0
            D1  = (1/m) * Delta1[:,1:] + (Lambda/m)*Theta1_reg # for j >= 1
            Theta1_grad = np.c_[D10, D1]

      grad = Theta1_grad.reshape((np.size(Theta1_grad),1))
      grad = np.vstack((grad,Theta2_grad.reshape(Theta2_grad.size,1)))

      return grad