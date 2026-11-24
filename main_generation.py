from PIL import Image
import numpy as np

from efros_leung import efros_leung


def main():
    input_dir = "textures_data/"
    input_paths = ["text0.png",
                   "text1.png",
                   "text2.png",
                   "text3.png",
                   "text4.png",
                   "text5.png",
                   "text6.png",
                   "text7.png",
                   "classification1.gif"]  
    
    output_dir = "exports/"

    for image in input_paths:
        for patch_size in [5, 9, 15]:
            for threshold in [0.3, 0.1, 0.01]:
                scale = 1.2         # the result will be twice as large as the sample
                # patch_size = 9    # size of the neighborhood window (odd number)
                # threshold = 0.1

                input_path = input_dir + image
                output_path = output_dir + image + '_' + str(patch_size) + '_' + str(threshold) + '.png'

                # load the sample texture as a float RGB array
                sample_image = Image.open(input_path).convert(mode='RGB')
                sample = np.array(sample_image).astype(float)

                # size of the texture to generate)
                height = int(np.round(sample.shape[0] * scale))
                width = int(np.round(sample.shape[1] * scale))

                # expand the texture with Efros-Leung
                result = efros_leung(sample, (height, width), patch_size, threshold)

                # convert back to an image and save it
                result = np.clip(result, 0, 255)
                result = np.round(result).astype(np.uint8)
                result_image = Image.fromarray(result)
                result_image.save(output_path)
                print("Saved expanded texture to", output_path)


if __name__ == "__main__":
    main()
