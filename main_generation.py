from PIL import Image
import numpy as np

from efros_leung import efros_leung


def main():
    input_path = "exports/text3_export.png"
    output_path = "exports/text3_expanded.png"
    scale = 1.2         # the result will be twice as large as the sample
    patch_size = 5    # size of the neighborhood window (odd number)
    threshold = 0.3

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
