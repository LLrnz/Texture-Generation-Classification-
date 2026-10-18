import numpy as np
import matplotlib.pyplot as plt


def mean_shift(data, labels, epsilon, window_size):
    shifted_x = data.copy()
    error = np.inf

    while error > epsilon:
        error = 0
        new_shifted_x = shifted_x.copy()

        for i, y in enumerate(shifted_x):

            distances = np.linalg.norm(data - y, axis=1)
            neighbors = data[distances < window_size]

            if len(neighbors) > 0:
                new_shifted_x[i] = np.mean(neighbors, axis=0)

            error = np.max(np.linalg.norm(new_shifted_x - shifted_x, axis=1))
            print(error)

        shifted_x = new_shifted_x
   
   
    centers = []

    for point in shifted_x:
        found = False

        for c, center in enumerate(centers):
            if np.linalg.norm(point - center) < window_size:
                labels.append(c)
                found = True
                break

        if not found:
            labels.append(len(centers))
            centers.append(point)
    return shifted_x


def show_classification(data, labels, cmap):
    x = data[:, 0]
    y = data[:, 1]

    fig, ax = plt.subplots()
    
    ax.scatter(x, y, c=labels,s=2, cmap=cmap)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()