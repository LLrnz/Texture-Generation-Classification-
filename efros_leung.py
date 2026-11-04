from typing import List, Tuple
import numpy as np
import random

Pixel = List[int] #rgb
Point = Tuple[int] #xy
Rect = Tuple[Point]



def get_patch(texture: np.ndarray, rect:Rect) -> np.ndarray:
    return texture[rect[0][0]:rect[1][0]+1, rect[0][1]:rect[1][1]+1]

def get_neighborhood_patch(texture: np.ndarray, p:Point, patch_size:np.ndarray) -> np.ndarray:
    patch_half_size = np.multiply(np.subtract(patch_size, 1), 0.5).astype(int) # //half the size of the patch TODO: better /!\
    padding_size = np.column_stack((patch_half_size, patch_half_size))
    half = patch_size[0] // 2
    
    padded_image = np.pad(texture, ((half, half), (half, half), (0, 0)), mode='constant', constant_values=0)
    return padded_image[p[0]:p[0]+patch_size[0], p[1]:p[1]+patch_size[1]]


def get_known_mask(heatmap: np.ndarray, p:Point, patch_size:int) -> np.ndarray:
    #window of the heatmap around p: 1 where the pixel of res is filled, 0 where it is unknown
    half = patch_size // 2
    padded_heatmap = np.pad(heatmap, half, mode='constant', constant_values=0) #outside of res = unknown
    window = padded_heatmap[p[0]:p[0]+patch_size, p[1]:p[1]+patch_size]
    return (window == -np.inf).astype(float)

def get_random_patch(texture: np.ndarray, patch_size:np.ndarray) -> np.ndarray:
    text_size = texture.shape
    # if patch_size[0] > text_size[0] or patch_size[1] > text_size[1]:
    #     patch_size = min(text_size[0], text_size[1])

    x = random.randint(0, text_size[0]-patch_size[0])
    y = random.randint(0, text_size[1]-patch_size[1])

    return get_patch(texture, ((x,y), (x+patch_size[0]-1, y+patch_size[1]-1)))

def copy_patch(texture: np.ndarray, patch: np.ndarray, rect:Rect) -> np.ndarray:
    texture[rect[0][0]:rect[1][0]+1, rect[0][1]:rect[1][1]+1] = patch
    return texture

def copy_patch_at_random(texture: np.ndarray, patch: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    texture_size = texture.shape
    patch_size = patch.shape
    x_size = patch_size[0]
    y_size = patch_size[1]

    x = random.randint(0, texture_size[0]-x_size)
    y = random.randint(0, texture_size[1]-y_size)

    texture[x:x+x_size, y:y+y_size] = patch
    update_heatmap(heatmap, ((x,y), (x+x_size-1, y+y_size-1)))
    return texture

def update_heatmap(heatmap: np.ndarray, rect:Rect) -> np.ndarray:
    heatmap[rect[0][0]:rect[1][0]+1, rect[0][1]:rect[1][1]+1] = np.negative(np.inf)
    #for all neighbors of the rect, input the number of filled neighbors in the heatmap
    # apply a mask on heatmap to select only cells of value:  np.negative(np.inf)

    # only the pixels touching the rect can change: scan the rect grown by 1 pixel on each side
    i_start = max(rect[0][0]-1, 0)
    i_end = min(rect[1][0]+2, heatmap.shape[0])
    j_start = max(rect[0][1]-1, 0)
    j_end = min(rect[1][1]+2, heatmap.shape[1])

    # #scan heat map with a 3x3 window and sum the number of filled neighbors for each pixel in the window
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            i_min = max(i-1, 0)
            i_max = min(i+2, heatmap.shape[0])
            j_min = max(j-1, 0)
            j_max = min(j+2, heatmap.shape[1])
            window = heatmap[i_min:i_max,j_min:j_max]

            if heatmap[i,j] != np.negative(np.inf):
                heatmap[i,j] = (window==np.negative(np.inf)).sum();
       
    return heatmap

def get_max_filled_neighborhood_points(heatmap: np.ndarray) -> List[Point]:
    return np.argwhere(heatmap == heatmap.max())

def pick_random_point(candidates: List[Point]) -> Point:
    return random.choice(candidates)

def place_pixel(pix:Pixel, point: Point, heatmap:np.ndarray, res_image:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    res_image[point[0]][point[1]] = pix
    update_heatmap(heatmap, (point, point))
    return (res_image, heatmap)


def compute_patch_distance(p1 : np.ndarray, p2:np.ndarray) -> float:
    #assuming array of float
    #assuming positive values
    distance = p1-p2
    distance = distance * distance
    return sum(sum(sum(distance)))

#for one pixel = max(heatmap), compute neighborhood distance with each sub image to find best
def compute_patch_distance_to_texture(nbh : np.ndarray, texture:np.ndarray, mask:np.ndarray) -> np.ndarray:
    #assuming array of float
    #assuming odd patch size
    
    patch_half_size = np.multiply(np.subtract(nbh.shape, 1), 0.5).astype(int) # //half the size of the patch TODO: better /!\
    padding_size = np.column_stack((patch_half_size, patch_half_size))
    padding_size[2] = [0,0]
    patch_half_size_x = patch_half_size[0]
    patch_half_size_y = patch_half_size[1]
    patch_half_size_z = 0
    
    padded_image = np.pad(texture, (padding_size), mode='constant', constant_values=(0))

    distance_map = np.full((texture.shape[0], texture.shape[1]), np.inf)

    #the center of nbh is not filled yet, so the mask already ignores it

    for i in range(patch_half_size_x, texture.shape[0]+patch_half_size_x):
        for j in range(patch_half_size_y, texture.shape[1]+patch_half_size_y):
            window = padded_image[i-patch_half_size_x:i+patch_half_size_x+1,j-patch_half_size_y:j+patch_half_size_y+1,:]
            diff = (nbh - window) ** 2
            diff = diff * mask[:, :, None]       # ignore unknown pixels of res
            distance = diff.sum()
            distance_map[i-patch_half_size_x][j-patch_half_size_y] = distance

    return np.round(distance_map).astype(int)
        
def select_best_patches(distance_map: np.ndarray, texture: np.ndarray, threshold: float) -> List[Pixel]:
    #keep every pixel whose distance is close to the best one: d <= d_min * (1 + threshold)
    min_distance = distance_map.min()
    candidates = (distance_map <= min_distance * (1 + threshold))
    return texture[candidates]

def pick_random_pixel(candidates: List[Pixel]) -> Pixel:
    return random.choice(candidates)

def efros_leung(sampled_texture: np.ndarray, size: tuple, patch_size: int = 5, threshold:float = 0.3 ) -> np.ndarray:

    # Init empty result texture
    res = np.zeros((size[0], size[1], 3))
    heatmap = np.zeros((size[0], size[1]))

    # Get random patch from sample texture
    array_patch_size = np.array([patch_size, patch_size, 3])
    patch = get_random_patch(sampled_texture, np.array([3,3]))

    # Paste patch at random position in result texture
    res = copy_patch_at_random(res, patch, heatmap)

    #While there are still unfilled pixels
    #==while heatmap has inf values
    while (heatmap != -np.inf).any():
        #pick a pixel with a max filled number
        not_filled_point = pick_random_pixel(get_max_filled_neighborhood_points(heatmap))
        candidate_neighborhood = get_neighborhood_patch(res, not_filled_point, array_patch_size)
        known_mask = get_known_mask(heatmap, not_filled_point, patch_size)

        #Compute distance to neighborhood
        distance_map = compute_patch_distance_to_texture(candidate_neighborhood, sampled_texture, known_mask)
        random_best_pixel= pick_random_pixel(select_best_patches(distance_map, sampled_texture, threshold))
        res, heatmap = place_pixel(random_best_pixel, not_filled_point, heatmap, res)


    return res


