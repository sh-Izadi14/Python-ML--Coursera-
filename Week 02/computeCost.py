
def computeCost(X, y, theta):
    m =  len(y)
    J = (1/(2*m)) * ( (X @ theta) - y).T @ ((X @ theta) - y) 
    return J
