import numpy as np

def generateColors(K, idx):
	generated_colors = np.zeros((K,3))
	if K ==3:
		generated_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
	else:
		for i in range(K):
			np.random.seed(i)
			generate_rgb = np.random.randint(0, 255, size=3)
			generated_colors[i, :] = generate_rgb
	colors = generated_colors[idx,:]
	
	return colors/255

