# img2mesh - a graphical front end for trussnet

img2mesh opens an image, runs [trussnet](../README.md) on it with the settings
you choose, and shows the image and the mesh together. You can crop both to an
x/y/z box and make them translucent.

- **Images:** NIfTI-1/2 (`.nii`, `.nii.gz`) and JNIfTI (`.jnii`, `.bnii`). These
  can be label volumes, gray-scale images or 4-D tissue-probability maps. A 4-D
  map is shown one channel at a time, or as the labels trussnet will mesh.
  That view follows trussnet's rules: channels named background, air, bg,
  outside, exterior or none are the exterior (hidden), with no such channel
  the exterior is 1 - sum(tissues), and the `--tpm-map`,
  `--tpm-exterior` and `--tpm-thresh` fields (Probability maps section) apply as
  you type them.
- **Meshes:** trussnet's `.jmsh` / `.bmsh`, tetrahedral or surface-only
  (`MeshTri` / `MeshSurf`, with or without labels), and `.off` / `.stl`
  surfaces, placed on the image by the image's own affine (sform or qform),
  so the mesh and the voxels line up. Open them from the toolbar or drop them
  on the window.
- **Modes:** the Mode section's "Make" choice is trussnet's `--mode`. `mesh`
  and `surface` make tets or the region surfaces of the image; `remesh`,
  `repair`, `cdt` and `optimize` take the mesh shown instead (a tet mesh gives
  its region surfaces), so the steps chain: open a surface, `repair` it, then
  `cdt` the result, then `optimize`. The section also holds the modes' own
  options (`--faces`, `--exact-tess`, `--raster-voxel`, `--cdt-fill`,
  `--opt-rounds`, and `--overlap` / `--auto-labels` for how the regions of a
  surface are found).
- **Layout:** as MCX Studio 2, a toolbar of large icons over their captions,
  and on the left an accordion of sections in two groups: Meshing (Mode, Sizing,
  Quality, Relaxation, Gray-scale input, Probability maps, Run, TrussNet path,
  Other arguments; blue titles) and Display (Crop box, Labels, Image, Mesh;
  teal titles). Click a section's title to open it; one is open at a time.
- **Settings:** every trussnet option, one field each. An empty field keeps
  trussnet's default, which is shown greyed in the field. The command line that
  will run is shown above the log.
- **View:**
  - the image is ray-cast (maximum intensity or accumulated), with a colour map,
    an opacity and a cut-off;
  - the mesh shows as a cut-out: the tetrahedra inside the crop box, opaque
    or translucent, with their edges (on by default) and per-label colours;
    each label can be hidden. With the whole box it is the mesh's outer
    surface and region interfaces; a smaller box exposes the elements where
    it cuts;
  - the crop box clips the image and the mesh together;
  - both are shown in millimetres, scaled by the image header's voxel size
    (NIfTI `pixdim`, JNIfTI `VoxelSize`);
  - graduated x/y/z axes along the box, as MCX Studio 2's, with the
    anatomical direction each points to (R/L, A/P, S/I, from the image's
    sform / qform or JNIfTI `Affine`) beside its letter.
- **Labels:** one checkbox list holds the labels of the mesh and of the
  image (a label volume, or a 4-D map's argmax view, named after its
  channels). Unticking a label hides it in both; Show all / Hide all reset
  the list.
- **Drag and drop:** drop images and meshes onto the window to open them. When
  both are dropped together, the image opens first, so the mesh lands on it.
- Drag with the left button to rotate, the right or middle button to pan, and
  the wheel to zoom.

## Building

img2mesh needs Lazarus (2.2 or newer, with FPC 3.2) and OpenGL 3.3.

```
make                        # bin/img2mesh
make LAZBUILD=/path/to/lazbuild WIDGETSET=qt5
```

### Remote displays and software rendering

Some displays offer no OpenGL visual: X2Go, NX, VNC, and `ssh -X` to a server
without GLX. On these, img2mesh renders offscreen through EGL, on the GPU if
EGL can reach one and otherwise on Mesa's software rasteriser (llvmpipe). It
then copies each frame into the window. The choice is made at start-up; to
force one, use `--gl` or the `IMG2MESH_GL` environment variable:

| `--gl` | How it draws |
|---|---|
| `auto` (default) | A GL window if the display has a GL visual, else `egl` |
| `glx` | Always a GL window |
| `egl` | Offscreen: the GPU, else software |
| `soft` | Offscreen, Mesa's software rasteriser only |

The log's first line shows which one is in use. The offscreen modes need
`libEGL` (glvnd, with Mesa's and/or the GPU driver's EGL vendor library).

It looks for `trussnet` next to itself, in `../build/`, `../bin/`, and then on
`PATH`. You can also point it at a binary in the "TrussNet path" section.

## Command line

```
img2mesh [image] [mesh] [--tn "trussnet options"] [--run]
         [--gl auto|glx|egl|soft] [--show volume|mesh|both] [--hide L1,L2,..] [--clip xlo,xhi,ylo,yhi,zlo,zhi] [--page N]
         [--screenshot out.png [--shot-size WxH]]
```

`mesh` is a `.jmsh`, `.bmsh`, `.off` or `.stl` file. `--run` runs trussnet at
start-up: on the image, or on the mesh with a mesh mode (`--tn "--mode cdt"`). `--screenshot` saves the view and exits,
after the run if there is one. That is how img2mesh is tested without a
display, e.g.:

```
xvfb-run -a bin/img2mesh head.nii.gz --tn "--size 4 --gpu" --run \
         --clip 0,0.5,0,1,0,1 --screenshot head.png
```

```
xvfb-run -a bin/img2mesh surf.off --tn "--mode cdt" --run --screenshot cdt.png
```

## Credits

- The OpenGL layer (`mcxgl.pas`: camera, shaders, volume ray-casting) and the
  JData readers (`mcxjd.pas`) come from
  [MCX Studio 2](https://github.com/fangq/mcx) (`mcxstudio2/`, same author,
  GPL-3.0-or-later). So do the toolbar icons (`icons/`, embedded as
  `i2micons.lrs`; `make icons` regenerates it). The build files follow MCX
  Studio 2's.
- The mesh display follows [MCX Cloud](https://github.com/fangq/mcx)'s
  JavaScript mesh preview (`mcxcloud/v2`) and iso2mesh: the "volface"
  surface (outer faces and region interfaces) of the selected elements.

License: GPL-3.0-or-later, as trussnet.
