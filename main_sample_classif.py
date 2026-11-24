from mean_shift import show_classification, mean_shift
import numpy as np

data = np.loadtxt("classif_data/gmm2d.asc")
# labels = np.zeros(len(data[:, 0]), dtype=int)
labels = []

shifted = mean_shift(data, labels, 0.01, 2)

show_classification(data,  labels, "Accent")