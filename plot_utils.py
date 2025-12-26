import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from fw import FWSeries, FWImagePair

def display_image(img, name='image', waitkey=0):
    im2=deepcopy(img)
    cimg = np.uint8(cv2.normalize(im2, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def display_cimg(cimg, name='image', waitkey=0):
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

def display_image_with_circles(img, circles, name='image', waitkey=0):
    cimg = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    np_circles = np.uint16(np.around(circles))
    for c in np_circles:
        cv2.circle(cimg,(c[0],c[1]),c[2],(0,255,0),2)
    cv2.imshow(name, cimg)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()



def plot_slice_values(fw_series:FWSeries, vert_lines=[], directory_path=''):
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
    filename = f"{fw_series.image_pairs[0].pdff.PatientName}_{fw_series.series_number_pdff}_mean_stddev.png"
    plt.savefig(os.path.join(directory_path, filename))
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