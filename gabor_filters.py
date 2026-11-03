import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def change_base(x, y, theta):
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = y * math.cos(theta) - x * math.sin(theta)
    return x_theta, y_theta

def gabor_function(x : np.ndarray, y:np.ndarray, theta:float, f:float, sigma:float):
    x_theta, y_theta = change_base(x,y,theta)
    gaussian = np.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2)
    wave = np.cos(2*math.pi*f*x_theta)

    #normalize so that the sum of all the values of the filter = 0
    k = (gaussian * wave).sum() / gaussian.sum()
    return gaussian * (wave - k)

def generate_gabor_filters():
    filters=[]
    scales =  [2,4,8,16] # wavelength in pixels
    thetas = [0, 30, 60, 90, 120, 150] 
    for s in scales:
        for theta in thetas:
            f = 1.0/s
            theta = math.radians(theta) 
            sigma = 0.56 * s # magic number for wavelength ratio

            radius = int(math.ceil(3 * sigma)) # stop at 3*sigma because the gaussian is nearly 0

            # generate matrices centered on (0, 0) to compute gabor for every pixel of the filter at once
            y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(float)
            M = gabor_function(x,y,theta,f,sigma)

            M = M / np.abs(M).sum() # normalize
            filters.append(M)

    return filters


#print M.max(), M.min()

# n = 5 #len(scales)
# m = 5 #len(thetas)
# filters = generate_gabor_filters()
# for i in range(n*m):
#     plt.subplot(n, m, i+1)
#     plt.imshow(filters[i], cmap = cm.gist_rainbow, origin='lower')

# #plt.imshow(M.T, origin='lower')
# #plt.colorbar()

# #plt.savefig('GaborFilter.png')
# plt.show()
