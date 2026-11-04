import numpy as np
import numpy.testing as nptest

from efros_leung import compute_patch_distance, compute_patch_distance_to_texture, copy_patch_at_random, get_max_filled_neighborhood_points, get_patch, get_random_patch, place_pixel, select_best_patches, update_heatmap


def test_get_patch():
    texture = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    rect = ((1, 1), (2, 2))  # This should extract the subarray [[5, 6], [8, 9]]
    patch = get_patch(texture, rect)
    expected_patch = np.array([[5, 6],
                                [8, 9]])
    assert np.array_equal(patch, expected_patch), f"Expected {expected_patch}, but got {patch}"

def test_get_random_patch():
    texture = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
    patch_size = np.array([3,3,0])
    patch = get_random_patch(texture, patch_size)
    expected_shape = (int(texture.shape[0] * patch_size[0]), int(texture.shape[1] * patch_size[1]))
    assert patch.shape == expected_shape, f"Expected shape {expected_shape}, but got {patch.shape}"

def test_copy_patch_at_random():
        texture = np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]])
        patch = np.array([[100, 200],
                          [300, 400]])
        modified_texture = copy_patch_at_random(texture.copy(), patch)
        # Check if the patch is present in the modified texture
        found_patch = False
        for i in range(modified_texture.shape[0] - patch.shape[0] + 1):
            for j in range(modified_texture.shape[1] - patch.shape[1] + 1):
                if np.array_equal(modified_texture[i:i+patch.shape[0], j:j+patch.shape[1]], patch):
                    found_patch = True
                    break
            if found_patch:
                break
        assert found_patch, "The patch was not found in the modified texture"

def test_update_heatmap():
    heatmap = np.zeros((5, 5))
    rect = ((1, 1), (2, 2))  # This should update the heatmap in the area of the rectangle
    updated_heatmap = update_heatmap(heatmap.copy(), rect)
    expected_heatmap = np.array([[1.,      2.,      2., 1., 0.],
                                 [2., -np.inf, -np.inf, 2., 0.],
                                 [2., -np.inf, -np.inf, 2., 0.],
                                 [1.,       2.,     2., 1., 0.],
                                 [0.,       0.,     0., 0., 0.]])
    assert np.array_equal(updated_heatmap, expected_heatmap), f"Expected {expected_heatmap}, but got {updated_heatmap}"

def test_compute_patch_distance():
    p1 = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    
    p2 = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    
    assert compute_patch_distance(p1, p2) == 4.0
    

def test_compute_patch_distance_edge():
    nbh = np.array([[[256.0, 0.0, 0.0], [0.0, 256.0, 0.0], [0.0, 0.0, 256.0]],
                    [[256.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[256.0, 0.0, 0.0], [0.0, 256.0, 0.0], [0.0, 0.0, 256.0]]])
    
    patch = np.array([[[128.0, 128.0, 128.0], [128.0, 128.0, 128.0], [128.0, 128.0, 128.0]],
                    [[128.0, 128.0, 128.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[128.0, 128.0, 128.0], [0.0, 256.0, 0.0], [0.0, 0.0, 256.0]]])
    expected_disance = (128*128)*3*5
    assert compute_patch_distance(nbh, patch) == expected_disance
    

def test_def_compute_patch_distance_to_image():
    patch = np.array([[[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]],
                    [[255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]]])
    
    image = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 255.0]],
                    [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0], [0.0, 0.0, 0.0]],
                    [[255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    


    expected_result = np.array([
                    [8, 5, 7, 4],
                    [4, 6, 6, 8],
                    [7, 4, 9, 6],
                    [4, 3, 4, 4]])

    actual_result = compute_patch_distance_to_texture(patch, image)

    nptest.assert_array_equal(actual_result, expected_result)


def test_select_best_patches():
    distances = np.array([
                    [8, 5, 7, 4],
                    [4, 6, 6, 8],
                    [7, 4, 9, 6],
                    [4, 3, 5, 5]])
    
    image = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 255.0]],
                    [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0], [0.0, 0.0, 0.0]],
                    [[255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    
    actual_result = select_best_patches(distances, image, 0.5)

    expected_result = [[0.0, 0.0, 0.0], [255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [255.0, 0.0, 0.0], [0.0, 0.0, 255.0]]

    nptest.assert_array_equal(actual_result.sort(), expected_result.sort())

def test_pick_max_filled_neighborhood_point():
    heatmap = np.array([[1.,      2.,      2., 1., 0.],
                        [2., -np.inf, -np.inf, 2., 0.],
                        [2., -np.inf, -np.inf, 2., 0.],
                        [1.,       2.,     2., 1., 0.],
                        [0.,       0.,     0., 0., 0.]])
    expected_result = [
                    [0, 1],
                    [0, 2],
                    [1, 0],
                    [2, 0],
                    [1, 3],
                    [2, 3],
                    [3, 1],
                    [3, 2]]
    
    actual_result = get_max_filled_neighborhood_points(heatmap)

    nptest.assert_array_equal(actual_result.sort(), expected_result.sort())


def test_place_pixel():
    heatmap = np.array([[1.,      1.,      1., 0.],
                        [2., -np.inf,      2., 0.],
                        [2., -np.inf,      2., 0.],
                        [1.,       1.,     1., 0.]])

    image = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 255.0]],
                    [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0], [0.0, 0.0, 0.0]],
                    [[255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    pixel = [128, 128, 128]

    point = (1,2)

    expected_heatmap = np.array([[1.,      2.,      2., 1.],
                                 [2., -np.inf, -np.inf, 1.],
                                 [2., -np.inf,      3., 1.],
                                 [1.,       1.,     1., 0.]])
    
    expected_image = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 255.0]],
                    [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [128, 128, 128], [0.0, 0.0, 0.0]],
                    [[255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [255.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    
    actual_image, actual_heatmap = place_pixel(pixel, point, heatmap, image)

    nptest.assert_array_equal(actual_heatmap, expected_heatmap)
    nptest.assert_array_equal(actual_image, expected_image)