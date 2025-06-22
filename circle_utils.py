import numpy as np 
import cv2
import copy

def circle_finder_gen(img:np.ndarray, minDist:float=0.01, param1:float=300, param2:float=10, minRadius:int=2, maxRadius:int=20):
    # # docstring of HoughCircles: 
    # # HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # HOUGH_GRADIENT_ALT is supposed to be more accurate but it doesn't find any circles
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist, param1=param1, param2=0.9, minRadius=minRadius, maxRadius=maxRadius)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=0.01, param1=300, param2=0.99)
    return circles


def circle_finder_water(img:np.ndarray, minDist:float=0.01, param1:float=300, param2:float=10, minRadius:int=3, maxRadius:int=50):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = circles[0, :]
    return circles

def circle_finder_pdff(img:np.ndarray, minDist:float=1, param1:float=20, param2:float=10, minRadius:int=3, maxRadius:int=200):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = circles[0, :]
    return circles

def circle_finder_general(img:np.ndarray, minDist:float=0.01, param1:float=300, param2:float=10, minRadius:int=2, maxRadius:int=200):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = circles[0, :]
    return circles


def overlay_circles(img:np.ndarray, circles:np.ndarray):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles:
        cv2.circle(cimg,(c[0],c[1]),c[2],(0,255,0),2)
    display_image(cimg, name='Overlay')

def display_image(img:np.ndarray, name='display_image', waitkey=0, normalize=False):
    if normalize:
        img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    # Display the image
    cv2.imshow(name, img)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def circles_to_rois(circles, roi_radius_px:float) -> np.ndarray:
    rois = copy.deepcopy(circles)
    roi_radius_px_npfloat = np.float32(roi_radius_px)
    for r in rois:
        r[2] = roi_radius_px_npfloat
    return rois


def sort_topleft_to_bottomright(coords:np.ndarray, precision=-1) -> np.ndarray:
    '''
    Sort the coordinates by their y values and then by their x values.
    precision is the number of decimal places to round to: 1 for tenths, 0 for ones, -1 for tens, etc.
    '''
    # Round the coordinates according to deprecision
    rough_coords = np.round(coords, precision)
    sorted_indices = np.lexsort((rough_coords[:,0], rough_coords[:,1]))
    # Use the sorted indices to reorder the original coordinates
    sorted_coords = coords[sorted_indices]
    return sorted_coords

def test_sort_topleft_to_bottomright():
    # Define the coordinates with a little noise
    coords = np.array([
        [10.2, 20.1],
        [30.3, 20.2],
        [50.1, 20.4],
        [10.1, 40.2],
        [30.2, 40.3],
        [50.4, 40.1],
        [10.3, 60.1],
        [30.1, 60.2],
        [50.2, 60.3],
        [10.4, 80.2],
        [30.4, 80.1],
        [50.3, 80.4]
    ])
    # Sort the coordinates
    random_coords = copy.deepcopy(coords)
    np.random.shuffle(random_coords)
    sorted_coords = sort_topleft_to_bottomright(random_coords)
    for i in range(len(sorted_coords)):
        assert sorted_coords[i][0] == coords[i][0]
        assert sorted_coords[i][1] == coords[i][1]

if __name__ == '__main__':
    test_sort_topleft_to_bottomright()