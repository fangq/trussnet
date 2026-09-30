# v2m - a graphical front end for v2mesh

v2m (volume to mesh) opens a volume, runs [v2mesh](../README.md) on it with
the settings you choose, and shows the volume and the mesh together. You can crop both to an
x/y/z box and make them translucent.

- **Volumes:** NIfTI-1/2 (`.nii`, `.nii.gz`) and JNIfTI (`.jnii`, `.bnii`). These
  can be label volumes, gray-scale volumes or 4-D tissue-probability maps. A 4-D
  map is shown one channel at a time, or as the labels v2mesh will mesh.
  That view follows v2mesh's rules: channels named background, air, bg,
  outside, exterior or none are the exterior (hidden), with no such channel
  the exterior is 1 - sum(tissues), and the `--tpm-map`,
  `--tpm-exterior` and `--tpm-thresh` fields (Probability maps section) apply as
  you type them.
- **2-D pictures:** PNG, BMP, JPEG, GIF, TIFF and PNM open as one slice, seen
  from above, and v2mesh meshes them in 2-D (triangles) from a one-slice NIfTI
  copy in the temporary folder (the log names it). A picture of at most 64
  colours is a label image: a gray one keeps its gray values as the labels, a
  coloured one numbers its colours 0, 1, .. from the darkest (0, the darkest,
  is the exterior). Any other picture is an intensity image (its luminance,
  0 .. 255): set `--thresholds` to mesh it.
- **Shapes:** a `.json` file of shape constructs (MCX `Shapes`, JMesh `Shape*` /
  `CSG*`; see the main README's "Shape input") opens as the input like a volume,
  into the Shapes panel, where the design is edited and drawn (as MCX Studio's
  Volume Designer does):
  - the constructs in a tree, in order (each overwrites the ones before, all
    cut to the first). A CSG object holds its boolean operation, and an
    operation (CSGUnion / CSGIntersect / CSGSubtract) its operands, which can be
    operations in turn; in a CSGSubtract the operands after the first are
    marked "taken away". Add has every MCX, JMesh and CSG construct: a shape
    added while an operation is selected (or its object, or one of its
    operands) goes into that operation, so a boolean is built by selecting and
    adding; a CSGObject (or an operation added on its own) starts as an
    object with an empty operation, selected, and the next Tag. Delete, Up /
    Down (the order of an operation's operands matters for CSGSubtract), New,
    Open and Save work on the selected node;
  - the selected construct's fields in a table: type a JSON value (`10`,
    `[30,30,30]`, `"text"`) and the drawing follows;
  - the drawing: each construct translucent in its Tag's colour, the selected
    one more opaque, in the domain's frame (the Grid, else the first object);
    slabs, layers and planes cut to the domain; what a CSGSubtract takes away
    drawn faint. It is a preview -- v2mesh's mesh is exact.
  - Run meshes the design: its file, or (edited or never saved) a copy in the
    temporary folder. The mesh then takes the drawing's place; select a
    construct to see the drawing again.
- **Meshes:** v2mesh's `.jmsh` / `.bmsh`, tetrahedral or surface-only
  (`MeshTri` / `MeshSurf`, with or without labels), and `.off` / `.stl`
  surfaces, placed on the image by the image's own affine (sform or qform),
  so the mesh and the voxels line up. Open them from the toolbar or drop them
  on the window.
- **CAD models and PLCs:** STEP (`.step` / `.stp`) and TetGen (`.poly` /
  `.smesh`) files are read by v2mesh itself (`--mode convert`, with the
  Element size as the tessellation's edge cap) and shown as a surface. The
  Mode switches to `cdt` if it was an image mode, and the mesh modes then read
  the file itself, so the Sizing settings apply when it is meshed.
- **Opening files:** the toolbar's Open takes any of these and tells them
  apart by suffix (the dialog also filters by kind). Its arrow opens a menu
  with one kind each: Volume, Mesh or surface, and Shapes, CAD model or PLC.
  Dropping files on the window works the same way.
- **Modes:** the Mode section's "Make" choice is v2mesh's `--mode`. `mesh`
  and `surface` make tets or the region surfaces of the image; `remesh`,
  `repair`, `cdt` and `optimize` take the mesh shown instead (a tet mesh gives
  its region surfaces), so the steps chain: open a surface, `repair` it, then
  `cdt` the result, then `optimize`. The section also holds the modes' own
  options (`--faces`, `--exact-tess`, `--raster-voxel`, `--cdt-fill`,
  `--opt-rounds`, and `--overlap` / `--auto-labels` for how the regions of a
  surface are found).
- **Layout:** as MCX Studio 2, a toolbar of large icons over their captions
  on top, and the command line and v2mesh's output at the bottom (drag the
  bar above them to resize). Between them, the 3-D view, with three panels
  floating over it: Meshing at the top left (v2mesh path, Mode, Sizing,
  Quality, Relaxation, Gray-scale input, Probability maps, Shapes (SDF), Run,
  Other arguments; blue), Display at the top right (Crop box, Labels, Volume,
  Mesh, Mesh Quality; teal) and Shapes at the bottom left (the shape designer;
  amber).
  - The panels hide themselves: each shows only its title until the pointer is
    on it, and folds back to it a moment after the pointer leaves. Its pin
    (the circle in its title: a ring while it auto-hides, filled when pinned)
    keeps it as it is set; click the pin again to let it hide.
  - Drag a panel's title bar to move it (it snaps to the view's edges); click
    the title bar to collapse the panel to it, or to open it again. Drag its
    edges or corners (the thin frame round it) to resize it. Its × hides it.
    A panel opened (by the pointer or a click) may move to fit in the view;
    folded again, it goes back to where you put it.
  - Display's Mesh Quality section: the shown elements' count, quality (min,
    5th percentile, median, mean, how many below 0.1) and size (min, median,
    max, total), with a histogram of each -- the quality 12 (3V)^(2/3) / sum
    l^2 of a tet (Joe-Liu) or 4 sqrt(3) A / sum l^2 of a triangle, 1 for a
    regular one, and the volume (a surface mesh: the area) on a log scale.
    Only the labels ticked in Display count. It is computed when the section
    is shown, so a large mesh costs nothing until you open it.
  - With nothing open, the view says how to start (and for a shape file, that
    Run meshes it: shapes have no preview).
  - The toolbar's View menu: Fit view; Reset view (the default view, framed);
    a tick for each panel (Meshing, Display, Shapes), to show or hide it; and
    Reset the panels, which puts them all back where they started.
  - Click a section's title to open it; one section of a panel is open at a
    time.
  - The panels follow the desktop's GTK theme, light or dark.
  - Where the panels are, their sizes, which are pinned, collapsed or hidden, is
    kept in `v2m.ini` in your configuration directory (`~/.config/v2m/` on
    Linux), at 96 dpi, so the layout fits a screen of any scaling. So is the
    folder of the last file opened or saved: the file dialogs start there.
  - Fit view frames the image or mesh in the part of the view between the
    panels.
  - Opening a file shows it from the front: from the image's anterior side
    (by its orientation), else from +y (the RAS convention), a little to the
    side and from above.
  - The window is a designed form, `i2mmain.lfm`: open `v2m.lpi` in the
    Lazarus IDE to edit it. Only the Meshing option rows are made at run time,
    from the option table in `i2mmain.pas`.
- **Settings:** every v2mesh option, one field each. An empty field keeps
  v2mesh's default, which is shown greyed in the field. The command line that
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
- Drag with the left button to rotate (about the centre of the axis box, also
  after a pan), the right or middle button (or Ctrl + left) to pan, and the
  wheel to zoom.

## Building

v2m needs Lazarus (2.2 or newer, with FPC 3.2) and OpenGL 3.3.

```
make                        # bin/v2m
make LAZBUILD=/path/to/lazbuild WIDGETSET=qt5
```

### Remote displays and software rendering

Some displays offer no OpenGL visual: X2Go, NX, VNC, and `ssh -X` to a server
without GLX. On these, v2m renders offscreen through EGL, on the GPU if
EGL can reach one and otherwise on Mesa's software rasteriser (llvmpipe). It
then copies each frame into the window. The choice is made at start-up; to
force one, use `--gl` or the `V2M_GL` environment variable:

| `--gl` | How it draws |
|---|---|
| `auto` (default) | A GL window if the display has a GL visual, else `egl` |
| `glx` | Always a GL window |
| `egl` | Offscreen: the GPU, else software |
| `soft` | Offscreen, Mesa's software rasteriser only |

The log's first line shows which one is in use. The offscreen modes need
`libEGL` (glvnd, with Mesa's and/or the GPU driver's EGL vendor library).

It looks for `v2mesh` next to itself, in `../build/`, `../bin/`, and then on
`PATH`. You can also point it at a binary in the "v2mesh path" section.

## Command line

```
v2m [image] [mesh] [--tn "v2mesh options"] [--run]
         [--gl auto|glx|egl|soft] [--show volume|mesh|both] [--hide L1,L2,..] [--clip xlo,xhi,ylo,yhi,zlo,zhi] [--page N]
         [--screenshot out.png [--shot-size WxH]]
```

`mesh` is a `.jmsh`, `.bmsh`, `.off` or `.stl` file. `--run` runs v2mesh at
start-up: on the image, or on the mesh with a mesh mode (`--tn "--mode cdt"`). `--page N` opens a
section: 1 the crop box, 2 Mesh Quality (else Mode). `--screenshot` saves the view and exits,
after the run if there is one. That is how v2m is tested without a
display, e.g.:

```
xvfb-run -a bin/v2m head.nii.gz --tn "--size 4 --gpu" --run \
         --clip 0,0.5,0,1,0,1 --screenshot head.png
```

```
xvfb-run -a bin/v2m surf.off --tn "--mode cdt" --run --screenshot cdt.png
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

License: GPL-3.0-or-later, as v2mesh.
