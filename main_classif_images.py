import random

from image_classification import classify

random.seed(0)
classify("classif_data/img", 64, max_images=5, patches_per_image=20, window_size=3)
