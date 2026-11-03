import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def ChangeBase(x,y,theta):
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = y * math.cos(theta) - x * math.sin(theta)
    return x_theta, y_theta

def GaborFunction(x,y,theta,f,sigma):
    r1 = ChangeBase(x,y,theta)[0] / sigma
    r2 = ChangeBase(x,y,theta)[1] / sigma
    arg = -0.5 * ( r1**2 + r2**2 ) / sigma**2
    return math.exp(arg) * math.cos(2*math.pi*f*(ChangeBase(x,y,theta)[0]/sigma))

def generate_gabor_filters():
    filters=[]  
    scales =  [2,4,8,16,32]
    thetas = [0, 30, 60, 90, 120]#[0, 30, 60, 90, 120, 150, 180]
    for s in scales:
        for theta in thetas:
            f = 1.0/s
            theta = math.radians(theta) # Converts angle x from degrees to radians.
            radius = 16
            sigma = 3

            M = np.zeros((radius*2,radius*2))

            x = -float(radius)
            for i in range(radius*2):
                y = -float(radius)
                for j in range(radius*2):
                    M[i,j] = GaborFunction(x,y,theta,f,sigma)
                    y = y + 1
                x = x + 1

            #print M.max(), M.min()
            # Normalization from 0 to 255
            #M[:,:] = ( (M[:,:] - M.min() ) * 255 ) / ( M.max() - M.min() )
            filters.append(M.T)

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
