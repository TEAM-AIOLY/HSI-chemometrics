import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA



main_data_folder = 'D:/data_ingi/trigo' 

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

idx=0
HSIreader.read_image(idx)
print(f"read image{HSIreader.current_name}")

#define wavelenrghts (for plots mostly)
wv =HSIreader.get_wavelength()
wv = [int(l) for l in wv]


hsi=HSIreader.current_image
n_rows, n_cols, n_channels =hsi.shape

n_samples =10000
# draw random samples and load them as a data array
x_idx = np.random.randint(0, n_cols, size=n_samples)
y_idx = np.random.randint(0, n_rows, size=n_samples)
spectral_samples = np.zeros((n_samples, n_channels), dtype=hsi.dtype)
coords = list(zip(y_idx, x_idx))
spectral_samples = np.array(HSIreader.extract_pixels(coords))

plt.figure()
for i in range(n_samples):
    plt.plot(wv, spectral_samples[i, :], label=f'Sample {i+1}' if i < 5 else "", alpha=0.6)
plt.xlabel("Wavelength")
plt.ylabel("Absorbance")
plt.title("Spectral samples")
plt.show(block=True)

pseudo_rgb= HSIreader.get_rgb()
plt.figure()
plt.imshow(pseudo_rgb)
plt.axis('off')
plt.show()

HSIreader.get_spectrum();


X =spectral_samples.copy()
n_components = 3  # Change to 3 for 3D visualization
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)

fig,ax =plt.subplots()
for k in range(n_components):
    ax.plot(wv, pca_loadings[:, k], label=f'PC{k+1}' )
ax.set_xlabel("Wavelength")
ax.set_ylabel("Loading")
ax.set_title("PCA Loadings")
ax.legend()
plt.show()