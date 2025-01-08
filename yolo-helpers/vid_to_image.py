import cv2
import imagej
import numpy as np

def create_partial_circular_mask(frame):
    '''
    Create mask to crop out black edges from microscope images
    Assume that the border is constant, just do this once
    '''

    print('Creating mask...')
    
    # Convert the frame to grayscale and apply a binary threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and take the two largest ones
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contours = sorted_contours[:4]  # Get the two largest contours

    # Create a mask and fill in the two largest contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, largest_contours, -1, 255, thickness=cv2.FILLED)
    inverted_mask = cv2.bitwise_not(mask)

    print('Mask created with the two largest contours.')

    return inverted_mask

def apply_mask_to_first_and_fifth_frame(video_path, mask, output_path_first, output_path_fifth):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Process the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from video.")
        return

    # Convert first frame to RGBA
    first_frame_rgba = cv2.cvtColor(first_frame, cv2.COLOR_BGR2BGRA)

    # Set alpha channel based on the mask
    first_frame_rgba[mask == 0, 3] = 0  # Transparent where mask is 0
    first_frame_rgba[mask != 0, 3] = 255  # Opaque where mask is not 0

    # Save the first frame with transparency
    cv2.imwrite(output_path_first, first_frame_rgba)
    print(f"Masked first frame with transparency saved as {output_path_first}")

    # Skip to the fifth frame
    for _ in range(4):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the fifth frame from video.")
            return

    # Convert fifth frame to RGBA
    fifth_frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Set alpha channel based on the mask
    fifth_frame_rgba[mask == 0, 3] = 0  # Transparent where mask is 0
    fifth_frame_rgba[mask != 0, 3] = 255  # Opaque where mask is not 0

    # Save the fifth frame with transparency
    cv2.imwrite(output_path_fifth, fifth_frame_rgba)
    print(f"Masked fifth frame with transparency saved as {output_path_fifth}")

    cap.release()

def stitch_video_to_image(video_path, mask, output_image_path, max_frames=3):
    print('Stitching video...')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    frame_count = 0
    print('Applying mask to frames...')
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the mask to each frame so only the region of interest is used
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert masked areas to black (optional, helps the stitcher focus on key areas)
        masked_frame[mask == 0] = (0, 255, 0)

        # Ensure the frame is in uint8 format
        if masked_frame.dtype != np.uint8:
            masked_frame = masked_frame.astype(np.uint8)

        frames.append(masked_frame)
        frame_count += 1
        print(f'Processed frame {frame_count}')

    cap.release()
    
    if len(frames) < 2:
        print("Error: Not enough frames for stitching.")
        return

    # Initialize the stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    print('Stitching frames...')
    status, stitched_image = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_image_path, stitched_image)
        print(f"Stitched image saved as {output_image_path}")
    else:
        print("Error: Could not stitch images. Stitcher status code:", status)

# Usage
video_path = "/Path/to/video/for/stitching"
output_first_frame_path = "/Output location 1"
output_fifth_frame_path = "/Output location 2"
output_image_path = "/Output location 3"

cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if ret:
    mask = create_partial_circular_mask(first_frame)
    cap.release()
    apply_mask_to_first_and_fifth_frame(video_path, mask, output_first_frame_path, output_fifth_frame_path)
    #stitch_video_to_image(video_path, mask, output_image_path)
else:
    print("Error: Could not read from video to create mask.")
