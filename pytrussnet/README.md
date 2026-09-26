# trussnet (Python)

GPU particle (truss) mesher: conforming tetrahedral meshes of multi-label, gray-scale
and tissue-probability volumes, and triangle meshes of 2-D images.

    import trussnet
    out = trussnet.tetmesh(vol, size=3, gpu=True)          # 3-D labels / gray-scale / 4-D TPM
    out = trussnet.tetmesh_file("head.nii.gz", size=3)     # a volume file, world coordinates
    out = trussnet.trimesh(img, size=3)                    # a 2-D image -> triangles
    node, elem, face = out["node"], out["elem"], out["face"]

`elem` holds 1-based node indices plus the label. The OpenCL GPU path falls back to the
CPU when no OpenCL device is available. See https://github.com/fangq/trussnet.
