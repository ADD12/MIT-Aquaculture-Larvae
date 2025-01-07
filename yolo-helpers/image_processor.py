from PIL import Image
import os
import glob

def crop_to_square(image):
    width, height = image.size
    print(f'Processing image with dimensions: {width}x{height}')  # Log dimensions

    if width == 1696 and height == 2544:
        top = 424
        bottom = height - 424
        cropped_image = image.crop((0, top, width, bottom))
    elif width == 2544 and height == 1696:
        left = 424
        right = width - 424
        cropped_image = image.crop((left, 0, right, height))
    else:
        print(f"Skipping image with unexpected dimensions: {width}x{height}")
        return None

    return cropped_image

def process_images(input_folder, output_folder):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Gather image paths
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.jpeg")) + glob.glob(os.path.join(input_folder, "*.JPG")) + glob.glob(os.path.join(input_folder, "*.JPEG"))

    # Check if any images are found
    if not image_paths:
        raise ValueError(f"No images found in the specified input folder: {input_folder}")

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        print(f"Output folder does not exist. Creating: {output_folder}")
        os.makedirs(output_folder)

    for image_path in image_paths:
        print(f'Processing file: {image_path}')
        try:
            with Image.open(image_path) as img:
                square_image = crop_to_square(img)
                if square_image is None:
                    continue  # Skip if dimensions are unexpected

                bw_image = square_image.convert("L")

                file_name = os.path.basename(image_path)
                save_path = os.path.join(output_folder, file_name)
                bw_image.save(save_path, "JPEG")
                print(f"Processed and saved: {save_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Example usage
input_folder = "/Users/beckettdevoe/Desktop/Larvae Main/100CANON"
output_folder = "/Users/beckettdevoe/Desktop/Larvae Main/square_images"

process_images(input_folder, output_folder)
