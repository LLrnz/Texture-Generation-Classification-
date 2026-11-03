import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

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


def show_image_classification(patches, data, labels, cmap):
    x = data[:, 0]
    y = data[:, 1]

    cmap = plt.get_cmap(cmap)
    nb_clusters = max(labels) + 1
    colors = [cmap(label / max(nb_clusters - 1, 1)) for label in labels]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors, s=2)

    for x0, y0, patch, color in zip(x, y, patches, colors):
        ab = AnnotationBbox(OffsetImage(patch.astype(np.uint8), zoom=0.3), (x0, y0),
                            frameon=True, pad=0.15, bboxprops=dict(edgecolor=color, facecolor=color, linewidth=2))
        ax.add_artist(ab)

    ax.set_title(f"{nb_clusters} clusters")

    plt.tight_layout()
    plt.show()
