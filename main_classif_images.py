import random

from image_classification import classify

random.seed(0)
classify("classif_data/exports", 32, max_images=14, patches_per_image=5, window_size=2.5, threshold=0.001)
