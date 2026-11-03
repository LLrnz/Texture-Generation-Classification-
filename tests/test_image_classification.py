import numpy as np
import numpy.testing as nptest
import pytest
from PIL import Image

from gabor_filters import generate_gabor_filters
from image_classification import assign_to_nearest_center, keep_images, sample_per_image, cluster_centers, cluster_to_image, gabor_features, image_splicer, well_classified_ratio, load_all_images_from_folder, splice_all_images


def save_gif(path, array):
    Image.fromarray(array.astype(np.uint8)).save(path)


def make_image(h, w):
    # Every pixel holds a unique value so patches can be traced back to their position
    values = np.arange(h * w * 3, dtype=float).reshape(h, w, 3)
    return values


@pytest.fixture
def gif_folder(tmp_path):
    save_gif(tmp_path / "a.gif", np.full((8, 6), 10))
    save_gif(tmp_path / "b.gif", np.full((4, 4), 200))
    save_gif(tmp_path / "ignored.png", np.full((5, 5), 50))
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def test_load_all_images_only_loads_gifs(gif_folder):
    images = load_all_images_from_folder(gif_folder)
    assert len(images) == 2


def test_load_all_images_returns_rgb_float_arrays(gif_folder):
    images = load_all_images_from_folder(gif_folder)
    shapes = sorted(im.shape for im in images)
    assert shapes == [(4, 4, 3), (8, 6, 3)]
    for im in images:
        assert isinstance(im, np.ndarray)
        assert im.dtype == float


def test_load_all_images_keeps_pixel_values(gif_folder):
    images = load_all_images_from_folder(gif_folder)
    values = sorted(np.unique(im).tolist() for im in images)
    assert values == [[10.0], [200.0]]


def test_load_all_images_empty_folder(tmp_path):
    assert load_all_images_from_folder(tmp_path) == []


def test_image_splicer_returns_two_halves():
    im = make_image(6, 6)
    first, second = image_splicer(im, np.array([2, 2]))
    assert len(first) + len(second) == 9
    assert abs(len(first) - len(second)) <= 1


def test_image_splicer_patch_shape():
    im = make_image(6, 9)
    first, second = image_splicer(im, np.array([2, 3]))
    for patch in first + second:
        assert patch.shape == (2, 3, 3)


def test_image_splicer_ignores_incomplete_border():
    # 7x5 image with 3x2 patches -> 2 rows * 2 cols of full patches
    im = make_image(7, 5)
    first, second = image_splicer(im, np.array([3, 2]))
    assert len(first) + len(second) == 4
    for patch in first + second:
        assert patch.shape == (3, 2, 3)


def test_image_splicer_patches_tile_the_image():
    im = make_image(4, 6)
    patch_size = np.array([2, 3])
    first, second = image_splicer(im, patch_size)

    expected = [im[i:i + 2, j:j + 3] for i in range(0, 4, 2) for j in range(0, 6, 3)]
    actual = first + second
    assert len(actual) == len(expected)
    for tile in expected:
        assert any(np.array_equal(tile, patch) for patch in actual), "Missing tile from the image"


def test_image_splicer_halves_are_disjoint():
    im = make_image(8, 8)
    first, second = image_splicer(im, np.array([2, 2]))
    for p in first:
        assert not any(np.array_equal(p, q) for q in second)


def test_image_splicer_shuffles(monkeypatch):
    im = make_image(4, 4)
    calls = []
    monkeypatch.setattr("image_classification.random.shuffle", lambda lst: calls.append(len(lst)) or lst.reverse())
    image_splicer(im, np.array([2, 2]))
    assert calls == [4]


def test_image_splicer_patch_larger_than_image():
    im = make_image(2, 2)
    first, second = image_splicer(im, np.array([3, 3]))
    assert first == [] and second == []


def test_splice_all_images(gif_folder):
    first, second = splice_all_images(gif_folder, np.array([2, 2]))
    # a.gif: 8x6 -> 4*3 = 12 patches (6 + 6), b.gif: 4x4 -> 2*2 = 4 patches (2 + 2)
    assert len(first) == 8
    assert len(second) == 8
    for image_id, patch in first + second:
        assert image_id in (0, 1)
        assert patch.shape == (2, 2, 3)


def test_splice_all_images_keeps_image_id(tmp_path):
    save_gif(tmp_path / "a.gif", np.full((4, 4), 10))
    save_gif(tmp_path / "b.gif", np.full((4, 4), 200))
    first, second = splice_all_images(tmp_path, np.array([2, 2]))
    images = load_all_images_from_folder(tmp_path)
    for image_id, patch in first + second:
        # every patch comes from a uniform image, so its color tells which image it belongs to
        assert np.all(patch == images[image_id][0, 0])


def test_gabor_features_shape():
    patches = [make_image(8, 8) for _ in range(3)]
    filters = generate_gabor_filters()
    assert gabor_features(patches, filters).shape == (3, len(filters) + 1)


def test_gabor_features_brightness_is_separate_from_texture():
    rng = np.random.default_rng(0)
    texture = rng.uniform(0, 100, size=(16, 16, 1)).repeat(3, axis=-1)
    dark, bright = gabor_features([texture, texture + 50], generate_gabor_filters())
    # same texture -> same gabor responses, only the brightness feature changes
    nptest.assert_allclose(dark[:-1], bright[:-1], atol=1e-6)
    assert bright[-1] - dark[-1] == pytest.approx(50)


def test_cluster_centers():
    shifted = np.array([[0.0, 0.0], [0.0, 2.0], [5.0, 5.0]])
    labels = np.array([0, 0, 1])
    nptest.assert_array_equal(cluster_centers(shifted, labels), [[0.0, 1.0], [5.0, 5.0]])


def test_assign_to_nearest_center():
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    features = np.array([[1.0, 1.0], [9.0, 8.0], [4.0, 4.0], [6.0, 6.0]])
    nptest.assert_array_equal(assign_to_nearest_center(features, centers), [0, 1, 0, 1])


def test_cluster_to_image_takes_majority():
    labels = np.array([0, 0, 0, 1, 1])
    ids = [3, 3, 1, 1, 1]
    nptest.assert_array_equal(cluster_to_image(labels, ids), [3, 1])


def test_well_classified_ratio():
    cluster_images = np.array([3, 1])
    labels = np.array([0, 0, 1, 1])
    ids = [3, 1, 1, 3]
    assert well_classified_ratio(labels, ids, cluster_images) == 0.5


def test_keep_images():
    patches = [(0, "a"), (1, "b"), (2, "c"), (1, "d")]
    assert keep_images(patches, [1, 2]) == [(1, "b"), (2, "c"), (1, "d")]


def test_sample_per_image():
    patches = [(0, i) for i in range(10)] + [(1, i) for i in range(3)]
    sampled = sample_per_image(patches, 5)
    ids = [image_id for image_id, _ in sampled]
    assert ids.count(0) == 5
    assert ids.count(1) == 3  # fewer patches than asked -> keep them all
    assert set(sampled) <= set(patches)
