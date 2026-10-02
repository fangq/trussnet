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
    cut to the first). A boolean operation (CSGUnion / CSGIntersect /
    CSGSubtract) shows with its Tag, its operands under it -- shapes, or
    operations in turn; in a CSGSubtract the operands after the first are
    marked "taken away". (In the file it is JMesh's CSGObject [operation,
    {Tag}], the one place a boolean carries its Tag.) Add has every MCX, JMesh
    and CSG construct: a shape added while an operation (or one of its
    operands) is selected goes into it, so a boolean is built by selecting and
    adding -- up to its two operands; a third becomes a new object at the end
    (nest an operation for more). An operation added on its own starts empty,
    selected, with the next Tag. A Grid inside an operation is the box [0, Size], as v2mesh reads
    it. Delete, Up / Down (the order of an operation's operands matters for
    CSGSubtract), New, Open and Save work on the selected node;
  - the selected construct's fields in a table: type a JSON value (`10`,
    `[30,30,30]`, `"text"`) and the drawing follows;
  - the drawing: each construct translucent in its Tag's colour, the selected
    one more opaque, in the domain's frame (the Grid, else the first object);
    slabs, layers and planes cut to the domain; what a CSGSubtract takes away
    drawn faint. It is a preview -- v2mesh's mesh is exact.
  - Run meshes the design: its file, or (edited or never saved) a copy in the
    temporary folder. The mesh then takes the drawing's place; select a
    construct to see the drawing again.
- **Meshes:** v2mesh's and brain2mesh's `.jmsh` / `.bmsh`, tetrahedral or
  surface-only, and `.off` / `.stl` surfaces. JMesh's typed containers are
  read as they are defined (`MeshVertex3`, `MeshTet4`, `MeshTri3`,
  `MeshQuad4`; no labels). In the flexible ones the last columns are labels:
  `MeshTri` / `MeshSurf` [m,4] a triangle and its label, [m,5] a triangle and
  the labels inside and outside, [m,6] a quad and its two labels
  (brain2mesh's SurfaceNets). brain2mesh's tissue shells (`--shells`: one
  object each, scalp to wm) become one surface, each shell a label named
  after its tissue. Nested arrays may be plain JSON or JData-annotated. All
  are placed on the image by the image's own affine (sform or qform),
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
  `cdt` the result, then `optimize`. With a shape design open, `cdt` meshes
  its exact surface instead, every crease, rim and knife edge kept (boxes,
  spheres, cylinders, cones, tori). The section also holds the modes' own
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
  - The main menu has every panel's commands: File (Open, Open volume / mesh
    / shapes, CAD or PLC, Save mesh, Save picture, Exit), Shapes (New, Open,
    Save design, Add any construct, Delete, Move up / down), Mesh (Run, Stop,
    the Make mode, each Meshing section), View (Fit and Reset view, show the
    volume / mesh / edges, Reset the crop box, each Display section, the
    panels) and Help (the guides and the issue tracker online, and About:
    the version, the license, the author, what v2m and v2mesh include and the
    works they build on). Short cuts: Ctrl+O open, Ctrl+S save the mesh,
    Ctrl+P save a picture, F5 run, Shift+F5 stop, Ctrl+F fit, F1 this guide,
    Ctrl+Q exit.
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
  - The view shows an image in RAS: its axes turned by its orientation
    (sform, qform, or a JNIfTI header's Orientation letters) so that R, A and
    S point right, to the front and up. The axis letters say where each image
    axis points. Opening a file shows it from the front, a little to the side
    and from above.
  - Display > Volume > Orientation overrides the header's orientation, for a
    file whose header has none or a wrong one. Pick or type three letters,
    one of R/L, A/P and S/I for x, y and z: PSL for a sagittal scan with
    slices from right to left, PSR for the other way. An Analyze 7.5 file
    converted by `savejnifti` is labelled RAS whatever the scan was. The
    override is for display only: v2mesh meshes in the header's orientation,
    and the image and mesh turn together. `--orient PSL` sets it at start-up.
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
- **brain2mesh and siamize panels** (hidden until View shows them): run
  [gpu_brain2mesh](https://github.com/NeuroJSON/gpu_brain2mesh)'s
  `brain2mesh` and [siamize](https://github.com/NeuroJSON/siamize) on the
  volume open, with every option of each in its own sections.
  - siamize segments a T1-weighted head MRI into SIAM's 18 classes (or SPM's 6,
    or a 4-D probability map). Its result, a binary JNIfTI whose label table
    names the classes, then becomes the volume shown: the input of brain2mesh
    and of v2mesh.
  - brain2mesh turns a label map (or a probability map) into SurfaceNets
    surfaces, closed tissue shells (`-S`) or tetrahedra (`--gpu-tet`), drawn
    on the image.
  - The steps chain: open a T1, Run siamize, then Run brain2mesh (or v2mesh's
    Run). Each panel shows its command and takes an Output file (empty: the
    temporary folder); the toolbar's Stop stops whichever is running.
  - The programs are found from `$V2M_BRAIN2MESH` / `$V2M_SIAMIZE` (a path),
    else the current folder, else v2m's own folder, else the PATH. One picked
    with a panel's "..." is kept in v2m.ini.
  - The Mesh menu runs them too (F6 brain2mesh, F7 siamize).
- **Labels:** one checkbox list holds the labels of the mesh and of the
  image (a label volume, or a 4-D map's argmax view, named after its
  channels). Unticking a label hides it in both; Show all / Hide all reset
  the list.
- **Drag and drop:** drop volumes and meshes onto the window to open them.
  Opening a volume replaces the mesh shown, and opening a mesh (or a CAD model
  or PLC) replaces the volume. A volume and a mesh dropped together, or given
  together on the command line, are both kept, the volume first, so the mesh
  lands on it; so is a run's result, drawn on its volume.
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
         [--gl auto|glx|egl|soft] [--show volume|mesh|both] [--hide L1,L2,..] [--clip xlo,xhi,ylo,yhi,zlo,zhi] [--orient PSL] [--page N]
         [--screenshot out.png [--shot-size WxH]]
         [--siam "siamize options"] [--siam-run] [--b2m "brain2mesh options"] [--b2m-run] [--panels b2m,siam]
```

`mesh` is a `.jmsh`, `.bmsh`, `.off` or `.stl` file. `--siam-run` /
`--b2m-run` run siamize / brain2mesh at start-up on the image (siamize first,
then brain2mesh on its segmentation, then v2mesh if `--run`); `--siam` /
`--b2m` give them more options, and `--panels` shows their panels. `--run` runs v2mesh at
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

```
xvfb-run -a bin/v2m t1.nii.gz --siam "-c opencl" --siam-run --b2m-run --screenshot head.png
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
