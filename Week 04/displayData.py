import numpy as np
import matplotlib.pyplot as plt

def displayData(X):
    
      m, n = X.shape
      example_width = int(np.round(np.sqrt(n)))
      example_height = int((n / example_width))

      # Compute number of items to display

      display_rows = int(np.floor(np.sqrt(m)))
      display_cols = int(np.ceil(m / display_rows))

      ## Between images padding
      pad = 1
      rows = pad + display_rows * (example_height + pad) # rows to display
      columns = pad + display_cols * (example_width + pad) # columns to display

      ## Setup blank display
      display_array = - np.ones((rows, columns))

      # Copy each example into a patch on the display array
      curr_ex = 0
      for j in range(display_rows):
            a = pad + j  * (example_height + pad) # row elements of display_array to substitute
            for i in range(display_cols):
            # Copy the patch
            # Get the max value of the patch
                  max_val = np.max(np.abs(X[curr_ex, :]))
                  b = pad + i  * (example_width + pad) # # column elements of display_array to substitute
                  
                  display_array[ a+0:a+example_height, b+0:b+example_width] = \
                                    X[curr_ex, :].reshape((example_height, example_width)) / max_val
                  curr_ex = curr_ex + 1
      # Display Image
      plt.imshow(display_array.T, cmap ='gray', origin = 'upper', extent=[-1, 1, -1, 1])
      plt.axis('off')
      plt.show()