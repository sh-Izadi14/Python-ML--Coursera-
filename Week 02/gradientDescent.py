import numpy as np
from computeCost import computeCost


def batchGD(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.ones((num_iters, 1))
    iter = 0
    
    while J_history[iter,0] >= 1e-7:
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta. 
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCostMulti) and gradient here.
    #
        theta = theta - (alpha/m) * ( X.T @ (X @ theta -y) ) 

    # ============================================================
    # Save the cost J in every iteration    
        
        J_history[iter,0] = computeCost(X, y, theta)

        if iter<num_iters-1:
           iter +=1
        else:
            break
    return theta, J_history, iter
