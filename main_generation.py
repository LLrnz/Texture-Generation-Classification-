from PIL import Image
import numpy as np

from efros_leung import efros_leung 

# load image
im_rgba = Image.open("textures_data/text3.png")
im = im_rgba.convert(mode='RGB')
im_array = np.array(im).astype(float)


# execute efro leung
result = efros_leung(im_array, (69, 66), 1)

result =  np.round(result).astype(np.uint8)
# build image
result_image = Image.fromarray(result)
result_image.save("exports/text3_export2.png")
