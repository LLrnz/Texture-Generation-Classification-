import math
import os
from typing import List, Tuple
from PIL import Image
import numpy as np
from efros_leung import get_patch
import random
import matplotlib.cm as cm


from gabor_filters import generate_gabor_filters
from mean_shift import mean_shift, show_image_classification

def splice_all_images(folder_path, patch_size) -> Tuple[List[Tuple[int, np.ndarray]]]:
	images = load_all_images_from_folder(folder_path)
	patchesDO = []
	patchesDT = [] 
	for image_id, im in enumerate(images):
		DO, DT = image_splicer(im, patch_size)
		patchesDO.extend((image_id, patch) for patch in DO)
		patchesDT.extend((image_id, patch) for patch in DT)
	return (patchesDO, patchesDT)

def load_all_images_from_folder(folder_path) -> List[np.ndarray] :
	image_list =[]
	for file in os.listdir(folder_path):
		if file.endswith(".png"):
			im = Image.open(os.path.join(folder_path, file))
			im = im.convert(mode='RGB')
			im = np.array(im).astype(float)
			image_list.append(im)

	return image_list

def image_splicer(im: np.ndarray, patch_size: np.ndarray) -> Tuple[List[np.ndarray]]:
	patches = []
	nb_patches = np.array(im.shape[:2]) // patch_size
	for i in range(nb_patches[0]):
		for j in range(nb_patches[1]):
			x1y1 = np.array([i,j]) * patch_size
			x2y2 = x1y1 + patch_size - 1
			patches.append(get_patch(im, (tuple(x1y1), tuple(x2y2))))

	random.shuffle(patches)
	half = len(patches)//2
	return (patches[:half], patches[half:])

def convolve_2d(patch: np.ndarray, filter: np.ndarray) -> np.ndarray:
	size_x, size_y = patch.shape # grayscale patch
	filter_size_x, filter_size_y = filter.shape

	flipped = filter[::-1, ::-1]
	top, left = (filter_size_x - 1) // 2, (filter_size_y - 1) // 2
	padded = np.pad(patch, ((filter_size_x - 1 - top, top), (filter_size_y - 1 - left, left)))

	result = np.zeros((size_x, size_y))
	for i in range(size_x):
		for j in range(size_y):
			# part of the patch under the filter when it is centered on (i, j)
			window = padded[i:i + filter_size_x, j:j + filter_size_y]
			result[i, j] = np.sum(window * flipped)
	return result

def convolve_2d_fft(patch: np.ndarray, filter: np.ndarray) -> np.ndarray:
	# conv(patch, filter) = IFFT( FFT(patch) * FFT(filter) )
	size_x, size_y = patch.shape # grayscale patch
	filter_size_x, filter_size_y = filter.shape

	full_x, full_y = size_x + filter_size_x - 1, size_y + filter_size_y - 1

	patch_hat = np.fft.rfft2(patch, s=(full_x, full_y))
	filter_hat = np.fft.rfft2(filter, s=(full_x, full_y))
	full = np.fft.irfft2(patch_hat * filter_hat, s=(full_x, full_y))

	top, left = (filter_size_x - 1) // 2, (filter_size_y - 1) // 2
	return full[top:top + size_x, left:left + size_y]

def gabor_features(patches: List[np.ndarray], filters: List[np.ndarray]) -> np.ndarray:
	gray = np.stack(patches).mean(axis=-1)
	brightness = gray.mean(axis=(1, 2))
	# equalize patch brightness
	gray = gray - brightness[:, None, None] 

	features = np.empty((len(patches), len(filters) + 1))

	for i, g in enumerate(gray):
		for k, filter in enumerate(filters):
			response = convolve_2d_fft(g, filter)
			features[i, k] = np.abs(response).mean()

	features[:, -1] = brightness
	return features

def keep_images(patches: List[Tuple[int, np.ndarray]], kept_ids: List[int]) -> List[Tuple[int, np.ndarray]]:
	kept_ids = set(kept_ids)
	return [(image_id, patch) for image_id, patch in patches if image_id in kept_ids]

def sample_per_image(patches: List[Tuple[int, np.ndarray]], n: int) -> List[Tuple[int, np.ndarray]]:
	# n random patches from each image (all of them if the image has fewer)
	by_image = {}
	for image_id, patch in patches:
		by_image.setdefault(image_id, []).append((image_id, patch))
	sampled = []
	for image_patches in by_image.values():
		sampled.extend(random.sample(image_patches, min(n, len(image_patches))))
	return sampled

def cluster_centers(shifted: np.ndarray, labels: np.ndarray) -> np.ndarray:
	# center of a cluster = mean of the converged points carrying its label -> (nb_clusters, nb_features)
	centers = []
	for c in range(labels.max() +1):
		centers.append(shifted[labels == c].mean(axis=0))
	
	return np.array(centers)

def assign_to_nearest_center(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
	distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
	return np.argmin(distances, axis=1)

def cluster_to_image(labels: np.ndarray, ids: List[int]) -> np.ndarray:
	# image id the majority of each cluster's patches come from
	ids = np.asarray(ids)
	count = []
	for c in range(labels.max() +1):
		count.append(np.bincount(ids[labels == c]).argmax())
	return np.array(count)

def well_classified_ratio(labels: np.ndarray, ids: List[int], cluster_images: np.ndarray) -> Tuple[float, float]:
	labels = np.asarray(labels)
	ids = np.asarray(ids)
	ratio = float(np.mean(cluster_images[labels] == ids))

	# for each image: 1 / number of distinct clusters its patches fall into
	# 1.0 = all its patches are in the same cluster, 0.5 = spread across 2 clusters, ...
	inverse_spread = [1 / len(np.unique(labels[ids == image_id])) for image_id in np.unique(ids)]
	cluster_unicity = float(np.mean(inverse_spread))

	return ratio, cluster_unicity

def project_2d(features: np.ndarray) -> np.ndarray:
	# PCA on two axis to display
	centered = features - features.mean(axis=0)
	_, _, vt = np.linalg.svd(centered, full_matrices=False)
	return centered @ vt[:2].T


	

#_______________________________________________________________________________________
def classify(folder_path, patch_size, window_size=2, max_images=None, patches_per_image=None):
	patchesDO, patchesDT = splice_all_images(folder_path, patch_size)

	# for quick trials only
	if max_images is not None:
		all_ids = sorted({image_id for image_id, _ in patchesDO})
		kept_ids = random.sample(all_ids, min(max_images, len(all_ids)))
		patchesDO = keep_images(patchesDO, kept_ids)
		patchesDT = keep_images(patchesDT, kept_ids)
	if patches_per_image is not None:
		patchesDO = sample_per_image(patchesDO, patches_per_image)
		patchesDT = sample_per_image(patchesDT, patches_per_image)

	idsDO, patchesDO = map(list, zip(*patchesDO))
	idsDT, patchesDT = map(list, zip(*patchesDT))

	# convolate patches with gabor filters
	gabor_filters = generate_gabor_filters()
	convolved_DO = gabor_features(patchesDO, gabor_filters)
	convolved_DT = gabor_features(patchesDT, gabor_filters)

	# normalize and center
	mean = convolved_DO.mean(axis=0)
	std = convolved_DO.std(axis=0)
	convolved_DO = (convolved_DO - mean) / std
	convolved_DT = (convolved_DT - mean) / std

	# calculate mean shift with first half
	labels = []
	shifted = mean_shift(convolved_DO, labels, 0.01, window_size)
	labels = np.array(labels)
	centers = cluster_centers(shifted, labels)

	# apply classification to second half
	labelsDT = assign_to_nearest_center(convolved_DT, centers)

	# evaluate: each cluster stands for the image most of its DO patches come from
	cluster_images = cluster_to_image(labels, idsDO)
	ratioDO, unicityDO = well_classified_ratio(labels, idsDO, cluster_images)
	ratioDT, unicityDT = well_classified_ratio(labelsDT, idsDT, cluster_images)
	print(f"{len(centers)} clusters for {len(set(idsDO))} images")
	print(f"well classified DO (mean shift): {ratioDO:.1%}, cluster unicity: {unicityDO:.2f}")
	print(f"well classified DT (nearest center): {ratioDT:.1%}, cluster unicity: {unicityDT:.2f}")

	show_image_classification(patchesDO, project_2d(convolved_DO), labels, cm.tab20)
	return ratioDO, ratioDT
