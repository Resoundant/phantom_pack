import numpy as np 
import cv2

SPHERE_RADIUS_MM = 100
SPHERE_VIAL_RADIUS_MM = 11
SPHERE_VIAL_SEP_MM = int(3.5*SPHERE_VIAL_RADIUS_MM)
IMG_RESOLUTION = 256
IMG_FOV_MM = 280

def simulate_sphere(randomize=False) -> tuple[np.ndarray, np.ndarray]:
    ''' 
    SImulate the Calimetrix sphereical cal phantom, with 12 vials
    arrayed in three rows of four.
    
    Parameters:
        randomize (bool): If True, move the vials around a bit
    '''
    # Create a blank image
    img_size = IMG_RESOLUTION
    img_fov = IMG_FOV_MM
    px_size = img_fov/img_size
    pdff = np.zeros((img_size, img_size), dtype=np.uint8)
    water = np.zeros((img_size, img_size), dtype=np.uint8)

    # fill with noise
    pdff  = add_gaussian_noise(pdff, mean=50, std_dev=20)
    water = add_gaussian_noise(water, mean=2, std_dev=2)

    # add spherical case
    sphere_radius = int(SPHERE_RADIUS_MM/px_size)
    cv2.circle(pdff,  (img_size//2, img_size//2), sphere_radius, 0, -1)
    cv2.circle(water, (img_size//2, img_size//2), sphere_radius, 100, -1)

    # Define the circle parameters
    num_rows = 3
    num_cols = 4
    circle_radius = int(SPHERE_VIAL_RADIUS_MM/px_size)
    circle_centers = int(SPHERE_VIAL_SEP_MM/px_size)
    if (num_cols*circle_centers + 2*circle_radius > img_size):
        print("WARNING - circles too large for image")

    fill_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    x_offset = img_size//2 - 2.0*(circle_centers)
    y_offset = img_size//2 - 1.5*(circle_centers)
    # Draw the circles
    for i in range(num_rows):
        for j in range(num_cols):
            x_rand = 0
            y_rand = 0
            if randomize:
                x_rand = int(abs(np.random.normal(0, 0.2*circle_radius)))
                y_rand = int(abs(np.random.normal(0, 0.2*circle_radius)))
            x = int(j * circle_centers + circle_centers // 2 + x_offset + x_rand)
            y = int(i * circle_centers + circle_centers // 2 + y_offset + y_rand)
            # fill vials
            cv2.circle(pdff,  (x, y), circle_radius, int(fill_values[i*num_cols+j])       , -1)
            cv2.circle(water, (x, y), circle_radius, int(100 - fill_values[i*num_cols+j]) , -1)
            # vial walls
            wall_thick = 2
            cv2.circle(pdff,  (x, y), circle_radius+wall_thick, 0, wall_thick)
            cv2.circle(water, (x, y), circle_radius+wall_thick, 0, wall_thick)
    return pdff, water

def add_gaussian_noise(image:np.ndarray, mean=0, std_dev=10):
    """
    Add Gaussian noise to an image.

    Args:
        image (numpy array): Input image.
        mean (float, optional): Mean of the Gaussian distribution. Defaults to 0.
        std_dev (float, optional): Standard deviation of the Gaussian distribution. Defaults to 10.

    Returns:
        numpy array: Noisy image.

    Example usage:
        image = np.zeros((100, 100), dtype=np.uint8)
        noisy_image = add_gaussian_noise(image)
    """
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + noise
    image_min = np.iinfo(image.dtype).min
    image_max = np.iinfo(image.dtype).max
    noisy_image = np.clip(noisy_image, image_min, image_max).astype(image.dtype)
    return noisy_image
