"""Write the gray-scale test volume of iso2mesh sample/demo_grayscale_ex1.m:
log10|g1 g2| + 10 of two Green's functions (sources at z = 20 and 60) on a
40 x 40 x 80 grid, 1 mm voxels, float32 NIfTI. The two singular source voxels
get the largest finite value.  usage: python3 tools/mkgray_jacobian.py out.nii.gz
"""
import sys
import numpy as np
import jdata as jd

xi, yi, zi = np.meshgrid(np.arange(1, 41), np.arange(1, 41), np.arange(1, 81), indexing="ij")
r1 = (xi - 20.0) ** 2 + (yi - 20.0) ** 2 + (zi - 20.0) ** 2  # (squared, as in the demo)
r2 = (xi - 20.0) ** 2 + (yi - 20.0) ** 2 + (zi - 60.0) ** 2
k = 10
with np.errstate(divide="ignore", invalid="ignore"):
    g12 = (np.exp(1j * k * r1) / (4 * np.pi * r1)) * (np.exp(1j * k * r2) / (4 * np.pi * r2))
    v = np.log10(np.abs(g12)) + 10
v[~np.isfinite(v)] = v[np.isfinite(v)].max()
v = v.astype(np.float32)
jd.savenifti(v, sys.argv[1])  # (jdata >= 0.9.5: the standard x-fastest NIfTI layout)
print(sys.argv[1], v.shape, float(v.min()), float(v.max()))
