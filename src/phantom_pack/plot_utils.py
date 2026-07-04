import os
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from .fw import FWSeries

_SCREEN_SIZE = None

def _get_screen_size():
    global _SCREEN_SIZE
    if _SCREEN_SIZE is not None:
        return _SCREEN_SIZE

    screen_size = None
    if os.name == 'nt':
        try:
            import ctypes
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            screen_size = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        except Exception:
            pass

    if screen_size is None:
        root = None
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        except Exception:
            screen_size = (1920, 1080)
        finally:
            if root is not None:
                root.destroy()

    _SCREEN_SIZE = screen_size
    return _SCREEN_SIZE

def display_image(img, name='image', waitkey=0):
    """Normalize a grayscale image and display it in an OpenCV window."""
    im2=deepcopy(img)
    cimg = np.uint8(cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def display_cimg(cimg, name='image', waitkey=0):
    """Display an already prepared color image in an OpenCV window."""
    img_height, img_width = cimg.shape[:2]
    screen_width, screen_height = _get_screen_size()
    scale = min(1.0, (screen_width * 0.9) / img_width, (screen_height * 0.9) / img_height)
    if scale < 1.0:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, int(img_width * scale), int(img_height * scale))
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def display_image_with_circles(img, circles, name='image', waitkey=0):
    """Display a normalized image with circle overlays drawn in green."""
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles:
        cv2.circle(cimg,(c[0],c[1]),c[2],(0,255,0),2)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()



def plot_slice_values(fw_series:FWSeries, vert_lines=[], directory_path=''):
    """Save per-slice mean/stddev and median PDFF plots for a series."""
    # each entry in this will will be the 5 vials  in a slice
    pdff_means = []
    pdff_medians = []
    pdff_stddevs = []
    slice_locations = []

    for img in fw_series.image_pairs:
        mystats = img.slice_stats()
        pdff_means.append(mystats['pdff_means'])
        pdff_medians.append(mystats['pdff_medians'])
        pdff_stddevs.append(mystats['pdff_stddevs'])
        slice_locations.append(img.location_full)
    #convert to np.array and transpose, so that each row is one roi loc across slices
    data_means = np.array(pdff_means).transpose()
    data_medians = np.array(pdff_means).transpose()
    data_stddevs = np.array(pdff_stddevs).transpose()


    for i in range(data_means.shape[0]):
        plt.errorbar(slice_locations, data_means[i],  yerr=data_stddevs[i], fmt='.', label=f"Mean {i}")
    if len(vert_lines) > 0:
        plt.axvline(x=vert_lines[0],  color='r', linestyle='--', linewidth=1) # vertical lines at edge of selected slices
        plt.axvline(x=vert_lines[-1], color='r', linestyle='--', linewidth=1)
    plt.title(f"PDFF Means +/- StdDev {fw_series.series_number_pdff} - {fw_series.series_description_pdff}")
    plt.xlabel("Slice Location")
    plt.ylabel("Mean PDFF")
    plt.legend()
    plt.grid(True)
    # plt.show()
    # filename = f"{fw_series.image_pairs[0].pdff_img.PatientName}_{fw_series.series_number_pdff}_mean_stddev.png"
    # plt.savefig(os.path.join(directory_path, filename))
    plt.close()

    for i in range(data_means.shape[0]):
        plt.plot(slice_locations, data_medians[i], '-x', label=f"Median {i}")
    if len(vert_lines) > 0:
        plt.axvline(x=vert_lines[0],  color='r', linestyle='--', linewidth=1) # vertical lines at edge of selected slices
        plt.axvline(x=vert_lines[-1], color='r', linestyle='--', linewidth=1)
    plt.title(f"Median PDFFs per slice {fw_series.series_number_pdff} - {fw_series.series_description_pdff}")
    plt.xlabel("Slice Location")
    plt.ylabel("Median PDFF")
    plt.legend()
    plt.grid(True)
    # plt.show()
    filename = f"{fw_series.image_pairs[0].pdff.PatientName}_{fw_series.series_number_pdff}_median.png"
    plt.savefig(os.path.join(directory_path, filename))
    plt.close()



def plot_image(img, name='image', waitkey=1):
    """Normalize and display a single image in an OpenCV window."""
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()


def plot_circles_list(img, circles, name='image', waitkey=1):
    """Display an image with circle overlays from a list-like circle array."""
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles:
        cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()


def plot_circles_ndarray(img, circles, name='image', waitkey=1):
    """Display an image with circle overlays from a HoughCircles-style array."""
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles[0,:]: # don't remembwer what packing needed this slice
        cv2.circle(cimg,(c[CX],c[CY]),c[CR],(0,255,0),2)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()


def plot_selected_image(img_data:dict, dest_filepath:str=None, display_image=False):
    """Save or display paired PDFF and water images with circle and ROI overlays."""
    # put rois onto pdff and water images
    cimg_water = np.uint8(cv2.normalize(img_data["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
    cimg_water = cv2.cvtColor(cimg_water, cv2.COLOR_GRAY2BGR)
    cimg_pdff = np.uint8(cv2.normalize(img_data["pdff"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
    cimg_pdff = cv2.cvtColor(cimg_pdff, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(img_data["circles"]))
    for c in np_circles:
        cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(0,255,0),1)
        cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,255,0),1)
    np_rois = np.uint16(np.around(img_data["rois"]))
    for c in np_rois:
        cv2.circle(cimg_pdff, (c[0],c[1]),c[2],(0,0,255),1)
        cv2.circle(cimg_water,(c[0],c[1]),c[2],(0,0,255),1)
    # canvas to have both pdff (left) and water (right) in one image
    height, width, channels = cimg_water.shape
    canvas = np.zeros(( width, 2 * height, channels), dtype=np.uint8)
    canvas[:height, :width] = cimg_pdff
    canvas[:height, width:] = cimg_water

    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Selected Image", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_array(img_pack_data:list[dict], dest_filepath:str=None,display_image=False):
    """Save or display a tiled array of water images with circle overlays."""
    cols = 5
    rows = np.uint8(np.ceil(len(img_pack_data) / cols))
    # Create a blank canvas to hold the images
    # cimg_setup = np.uint8(img_pack_data[0]["water"].pixel_array)
    cimg_setup = cv2.cvtColor(np.uint8(img_pack_data[0]["water"].pixel_array), cv2.COLOR_GRAY2BGR) #bug: assumes all images same resolution
    height, width, channels = cimg_setup.shape

    canvas = np.zeros((height * rows, width * cols, channels), dtype=np.uint8)
    for i, mydict in enumerate(img_pack_data):
        cimg = np.uint8(cv2.normalize(mydict["water"].pixel_array, None, 0, 255, cv2.NORM_MINMAX))
        cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
        if mydict["circles"] is not None:
            np_circles = np.uint16(np.around(mydict["circles"]))
            for c in np_circles:
                cv2.circle(cimg,(c[0],c[1]),c[2],(255,255,0),1)             # draw the outer circle

        # Place each image on the canvas
        row = i // cols
        col = i % cols
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = cimg
        # Display the canvas

    # save image
    if (dest_filepath != None):
        cv2.imwrite(dest_filepath, canvas)

    if (display_image):
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
