import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA

from skimage.filters import threshold_multiotsu
from skimage.measure import label
from skimage.measure import regionprops
import json



main_data_folder = 'D:/viniot/' 
n_samples =10000

dataset =HsiDataset(main_data_folder,data_ext='dat')
nb_images = len(dataset)

if os.path.isdir(main_data_folder):
    if nb_images>0:
        print(f"dataset  is valid and contains {nb_images} image(s)")
    else:
        print('empty dataset')
else:
    print('path invalid')


HSIreader = HsiReader(dataset)

for idx in range(nb_images):
    HSIreader.read_image(idx)
    print(f"read image{HSIreader.current_name}")
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]

    hsi=HSIreader.current_image
    n_rows, n_cols, n_channels =hsi.shape


    x_idx = np.random.randint(0, n_cols, size=n_samples)
    y_idx = np.random.randint(0, n_rows, size=n_samples)
    spectral_samples = np.zeros((n_samples, n_channels), dtype=hsi.dtype)
    coords = list(zip(y_idx, x_idx))
    spectral_samples = np.array(HSIreader.extract_pixels(coords))

    X =spectral_samples.copy()
    n_components = 3  # Change to 3 for 3D visualization
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    score_img = HSIreader.project_pca_scores(pca_loadings)
    
    score_pc_ref = score_img[:,:,0]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    labeled_image = label(segmented)
    
    regions = regionprops(labeled_image)

    if len(regions) == 0:
        raise ValueError("No labeled objects found.")
    background_region = max(regions, key=lambda r: r.area)
    background_label = background_region.label
    
    main_region = max(regions, key=lambda r: r.area)
    main_label = main_region.label

    # Binary masks
    main_mask = (labeled_image == main_label).astype(np.uint8)
    discarded_mask = (labeled_image != main_label).astype(np.uint8) 
    
    image_data_path = HSIreader.dataset[HSIreader.current_idx]["data"]
    image_folder = os.path.dirname(image_data_path)
    image_name = HSIreader.current_name
    mask_path = os.path.join(image_folder, f"{image_name}_main_mask.npy")
    np.save(mask_path, main_mask)
    
    png_path = os.path.join(image_folder, f"{image_name}_main_mask.png")
    plt.imsave(png_path, main_mask, cmap='gray')
    # -------------------------------------------------------------------
    # Save main object properties
    # -------------------------------------------------------------------
    props_dict = {
        "label": main_region.label,
        "area": main_region.area,
        "bbox": main_region.bbox,           # (min_row, min_col, max_row, max_col)
        "centroid": main_region.centroid,
        "eccentricity": main_region.eccentricity,
        "perimeter": main_region.perimeter,
        "convex_area": main_region.convex_area
    }

    props_path = os.path.join(image_folder, f"{image_name}_main_properties.json")
    with open(props_path, "w") as f:
        json.dump(props_dict, f, indent=2)

    # -------------------------------------------------------------------
    # Visualization: main object = BLUE, discarded = RED, background = BLACK
    # -------------------------------------------------------------------
    # h, w = labeled_image.shape
    # vis = np.zeros((h, w, 3), dtype=np.uint8)  # default = black

    # vis[main_mask == 1] = [0, 0, 255]          # main object = BLUE
    # vis[discarded_mask == 1] = [255, 0, 0]     # discarded = RED

    # plt.figure(figsize=(8, 8))
    # plt.imshow(vis)
    # plt.title("Main object = BLUE, Discarded = RED")
    # plt.axis("off")
    # plt.show()