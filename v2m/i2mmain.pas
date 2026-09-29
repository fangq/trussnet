{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mmain -- the window: open an image, set v2mesh's options, run it, look at
  the image and the mesh together, cropped and translucent.

  The window is a designed form (i2mmain.lfm): the toolbar on top, the
  command line and log at the bottom, and the view between them, with two
  panels ("cards") floating over it: Meshing (top left) and Display (top
  right) -- drag a title to move one, its chevron collapses it, x hides it
  (the toolbar's View menu brings it back); the layout is kept in v2m.ini. The Meshing option rows are
  made at run time from a table of v2mesh's options (Options below), each in
  the designed section of its group, so a new v2mesh flag is one line here.
  An empty field is v2mesh's own default, which is shown greyed in the field. }
unit i2mmain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, StrUtils, Process, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, ComCtrls, CheckLst, Buttons, LCLType, LCLIntf, ImgList, Menus, IniFiles, ValEdit, fpjson, mcxgl, i2mvol,
  i2mmesh, i2mview, i2micons, i2mshapes;

type
  TI2MOptKind = (okFloat, okInt, okText, okBool, okChoice, okFlagArg);

  TI2MOption = record
    Flag: string;      { the v2mesh flag }
    Caption: string;
    Kind: TI2MOptKind;
    Default: string;   { shown greyed; for a choice, the items, '|'-separated }
    Hint: string;
    Group: string;     { a heading starts a new group }
  end;

  TI2MPanels = array of TPanel;

  TI2MMainForm = class(TForm)
  published
    { the designed form (i2mmain.lfm) }
    ActionBar, ShapeBar: TToolBar;
    BtnOpen, BtnMesh, ActionDiv1, BtnRun, BtnStop, ActionDiv2, BtnSave, BtnShot, ActionDiv3,
    BtnView, ShapeAddBtn, ShapeDiv0, ShapeUpBtn, ShapeDownBtn, ShapeDiv1, ShapeNewBtn,
    ShapeOpenBtn, ShapeSaveBtn, ShapeDiv2, ShapeDelBtn: TToolButton;
    StatusBar: TStatusBar;
    LogPanel, ViewHost, MeshingCard, MeshingTitle, SectPathHead, SectPathBody, ExeRow, FormatRow,
    SectModeHead, SectModeBody, SectSizingHead, SectSizingBody, SectQualityHead, SectQualityBody,
    SectRelaxHead, SectRelaxBody, SectGrayHead, SectGrayBody, SectTpmHead, SectTpmBody,
    SectShapesHead, SectShapesBody, SectRunHead, SectRunBody, SectOtherHead, SectOtherBody,
    DisplayCard, DisplayTitle, SectCropHead, SectCropBody, SectLabelsHead, SectLabelsBody,
    LabelButtons, SectImageHead, SectImageBody, SectMeshHead, SectMeshBody, SectStatsHead,
    SectStatsBody, ShapesCard, ShapesTitle, ShapesBody, EmptyHint: TPanel;
    CmdEdit, ExeEdit, ExtraEdit: TEdit;
    LogMemo: TMemo;
    LogSplitter, ShapeSplit: TSplitter;
    MeshingChevron, MeshingClose, MeshingCaption, ExeLabel, FormatLabel, DisplayChevron,
    DisplayClose, DisplayCaption, ClipLabel0, ClipLabel1, ClipLabel2, ClipLabel3, ClipLabel4,
    ClipLabel5, ChannelLabel, MapLabel, StyleLabel, OpacityLabel, FloorLabel, MeshAlphaLabel,
    StatsText, QualityCaption, SizeCaption, ShapesChevron, ShapesClose, ShapesCaption, ShapeHint,
    EmptyHintText: TLabel;
    MeshingBody, DisplayBody: TScrollBox;
    ExeBrowse: TButton;
    FormatCombo, ChannelCombo, MapCombo, StyleCombo: TComboBox;
    ClipXFrom, ClipXTo, ClipYFrom, ClipYTo, ClipZFrom, ClipZTo, OpacityTrack, FloorTrack,
    MeshAlphaTrack: TTrackBar;
    ResetClipButton, ShowAllButton, HideAllButton: TBitBtn;
    LabelList: TCheckListBox;
    ShowVolCheck, ShowMeshCheck, ShowEdgesCheck: TCheckBox;
    QualityHist, SizeHist: TPaintBox;
    ShapeTree: TTreeView;
    ShapeFields: TValueListEditor;
    ActionIcons: TImageList;
    ViewMenu, ShapeAddMenu: TPopupMenu;
    MenuFit, MenuResetView, MenuSep1, MenuMeshing, MenuDisplay, MenuShapes, MenuSep2, MenuReset,
    ShapeAddMCX, ShapeAdd_Grid, ShapeAdd_Box, ShapeAdd_Subgrid, ShapeAdd_Sphere, ShapeAdd_Cylinder,
    ShapeAdd_XSlabs, ShapeAdd_YSlabs, ShapeAdd_ZSlabs, ShapeAdd_XLayers, ShapeAdd_YLayers,
    ShapeAdd_ZLayers, ShapeAddJMesh, ShapeAdd_ShapeBox3, ShapeAdd_ShapeSphere,
    ShapeAdd_ShapeCylinder, ShapeAdd_ShapeCone, ShapeAdd_ShapeConeFrustum, ShapeAdd_ShapeEllipsoid,
    ShapeAdd_ShapeTorus, ShapeAdd_ShapeSphereShell, ShapeAdd_ShapeSphereSegment,
    ShapeAdd_ShapePlane3: TMenuItem;
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormDropFiles(Sender: TObject; const FileNames: array of string);
    procedure OptionChanged(Sender: TObject);
    procedure DisplayChanged(Sender: TObject);
    procedure ClipChanged(Sender: TObject);
    procedure LabelsChanged(Sender: TObject);
    procedure ChannelChanged(Sender: TObject);
    procedure OpenClick(Sender: TObject);
    procedure RunClick(Sender: TObject);
    procedure StopClick(Sender: TObject);
    procedure MeshClick(Sender: TObject);
    procedure SaveClick(Sender: TObject);
    procedure ShotClick(Sender: TObject);
    procedure ResetClipClick(Sender: TObject);
    procedure BrowseExeClick(Sender: TObject);
    procedure AllLabelsClick(Sender: TObject);
    { the cards and their sections }
    procedure CardTitleMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure CardTitleMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure CardTitleMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure CardCollapseClick(Sender: TObject);
    procedure CardCloseClick(Sender: TObject);
    procedure CardEdgeMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure CardEdgeMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure CardEdgeMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure SectionHeadClick(Sender: TObject);
    procedure SectionHeadEnter(Sender: TObject);
    procedure SectionHeadLeave(Sender: TObject);
    procedure ViewMenuClick(Sender: TObject);
    procedure BtnViewClick(Sender: TObject);
    procedure ViewHostResize(Sender: TObject);
    procedure HistPaint(Sender: TObject);
    { the Shapes panel }
    procedure ShapeAddBtnClick(Sender: TObject);
    procedure ShapeAddClick(Sender: TObject);
    procedure ShapeDelClick(Sender: TObject);
    procedure ShapeUpClick(Sender: TObject);
    procedure ShapeDownClick(Sender: TObject);
    procedure ShapeNewClick(Sender: TObject);
    procedure ShapeOpenClick(Sender: TObject);
    procedure ShapeSaveClick(Sender: TObject);
    procedure ShapeTreeChange(Sender: TObject);
    procedure ShapeFieldValidate(Sender: TObject; ACol, ARow: Integer; const OldValue: string; var NewValue: string);
  private
    { meshing options (the rows made from Options) }
    FEdits: array of TControl;   { per option: TEdit / TCheckBox / TComboBox }
    FArgEdits: array of TEdit;   { okFlagArg: the argument }
    FClip: array[0..5] of TTrackBar;   { the crop sliders, x from .. z to }
    { the cards: the one being dragged, and the designed layout (Reset layout) }
    FDragCard: TPanel;
    FDragFrom, FDragOrigin: TPoint;
    { the card being resized, by which edges (EdgeLeft or ..), from where }
    FSizeCard: TPanel;
    FSizeEdges: Integer;
    FSizeFrom: TPoint;
    FSizeOrigin: TRect;
    FDefaults: array of TRect;
    FEmptyText: string;   { the empty view's hint, as designed }
    FKeepQueued: Boolean; { the cards to be kept in the resized view (queued) }
    FLastDir: string;     { the folder of the last file opened or saved (kept in v2m.ini) }
    { the shape constructs being designed (the Shapes panel); FShapesOn: they
      are Run's input -- written to FShapeTemp first when edited or unsaved }
    FShapes: TI2MShapeDoc;
    FShapesOn, FShapesDirty: Boolean;
    FShapeSel: Integer;
    FShapeTemp: string;
    FShapeFitted: Boolean;   { the view framed on the design since it was opened }
    { the mesh's statistics: 40-bin histograms of the shown elements' quality
      (0..1) and size (log10, FSizeLo..FSizeHi) }
    FQualHist, FSizeHist: array of Integer;
    FSizeLo, FSizeHi: Double;
    FStatsDirty: Boolean;   { the mesh or its shown labels changed since they were computed }
    { state }
    FView: TI2MView;
    FVol: TI2MVolume;
    FVolFile, FMeshFile, FOutFile: string;
    FMesh: TI2MMesh;
    FProc: TProcess;
    FTimer: TTimer;
    FClipTimer: TTimer;   { a crop slider being dragged: one cut-out per pause }
    FPending: string;
    FStarted: TDateTime;
    FUpdating: Boolean;
    FEcho: Boolean;
    FArgmaxKey: string;   { the options the argmax view was made with }
    { the image shown: a channel of FVol, or FArgmax }
    FArgmax: TI2MSingles;
    FDispPtr: PSingle;
    FDispLo, FDispHi: Single;
    FDispIsLabel: Boolean;       { integer labels: the label list applies }
    FVolLabels: array of Boolean;   { labels present in it }
    FLabelNames: array of string;   { per label, for the list }
    FOrder: Integer;
    { increasing positions: aligned controls keep the order they were made in
      (equal ones, before the window is laid out, come out reversed) }
    function Next: Integer;
    procedure BuildMeshingSections;
    procedure Log(const AText: string);
    procedure ViewLog(Sender: TObject; const AText: string);
    procedure Poll(Sender: TObject);
    procedure Drain;
    procedure Finished;
    function Arguments(out AList: TStringList): Boolean;
    procedure UpdateCommand;
    procedure UpdateButtons;
    procedure ShowChannel;
    procedure FillLabels;
    procedure UploadVolume;
    procedure ScanVolumeLabels;
    function ChannelName(AChannel: Integer): string;
    function OptionText(const AFlag: string): string;
    function ArgmaxKey: string;
    function VoxelSize: TMcxVec3;
    procedure ClipTimer(Sender: TObject);
    function FindV2mesh: string;
    { --mode, from the Make choice; its input is the mesh shown (not the image) }
    function Mode: string;
    function MeshInput: Boolean;
    function InputMeshFile: string;
    { the cards }
    function AllCards: TI2MPanels;
    function CardOf(AControl: TControl): TPanel;
    function CardBody(ACard: TPanel): TControl;
    procedure SetCardCollapsed(ACard: TPanel; ACollapsed: Boolean);
    procedure KeepInView(ACard: TPanel; ASnap: Boolean = False);
    function SectionBody(AHead: TPanel): TPanel;
    function SectionHead(const ACaption: string): TPanel;
    procedure OpenHead(AHead: TPanel);
    procedure PaintHead(AHead: TPanel; AHot: Boolean);
    procedure UpdatePanelsMenu;
    procedure UpdateStats;
    procedure KeepCardsInView(Data: PtrInt);
    procedure UseLastDir(D: TFileDialog);
    procedure ShapesInput;
    procedure RefreshShapes;
    procedure FillShapeFields;
    procedure PreviewShapes;
    procedure ShapesChanged;
    procedure RememberDir(const AFileName: string);
    procedure CheckStats;
    procedure ComputeStats;
    procedure LoadLayout;
    procedure SaveLayout;
    function LayoutFile: string;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    function LoadImage(const AFileName: string): Boolean;
    { AReset: a file opened (not a run's result, nor the mesh moved onto a new
      image): the crop box back to the whole frame and the view refitted }
    function LoadMesh(const AFileName: string; AReset: Boolean = True): Boolean;
    procedure ResetBox;
    { -q 2 --size 3 ...: for the command line and scripted runs }
    procedure SetOption(const AFlag, AValue: string);
    procedure SetClip(const ALo, AHi: TMcxVec3);
    function SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
    procedure Run;
    function Running: Boolean;
    function Busy: Boolean;   { running, or its output not yet taken in }
    { AMargin: framed in what the cards leave free (not for a screenshot,
      which has no cards) }
    procedure FitView(AMargin: Boolean = True);
    { the default view: from the front (the image's anterior side; a mesh's +y) }
    procedure ResetView;
    { opens a section: 0 Mode, 1 the crop box, 2 the mesh quality }
    procedure ShowPage(AIndex: Integer);
    procedure OpenSection(const ACaption: string);
    { unticks these labels (comma-separated) }
    procedure HideLabels(const AList: string);
    { 'volume', 'mesh' or 'both' }
    procedure ShowOnly(const AWhat: string);
    property EchoLog: Boolean read FEcho write FEcho;
  end;

var
  I2MMainForm: TI2MMainForm;
  { the view's GL: auto / glx / egl / soft (see TI2MView.Create) }
  I2MGLMode: string = 'auto';

implementation

{$R *.lfm}

const
  Options: array[0..41] of TI2MOption = (
    (Flag: '--mode'; Caption: 'Make'; Kind: okChoice;
     Default: 'mesh: tets of the image|surface: the image''s surfaces|remesh: tets of the mesh''s surfaces|' +
       'repair: clean surfaces of the mesh|cdt: tets keeping the mesh''s surfaces|optimize: better tets of the mesh';
     Hint: 'what to make of what: the image (mesh, surface) or the mesh shown (remesh, repair, cdt, optimize; ' +
       'a tet mesh gives its region surfaces)'; Group: 'Mode'),
    (Flag: '--faces'; Caption: 'Also write the region surfaces'; Kind: okBool; Default: '';
     Hint: 'tets and their region surfaces (MeshTri) together'; Group: ''),
    (Flag: '--exact-tess'; Caption: 'Surfaces: tessellate every node'; Kind: okBool; Default: '';
     Hint: 'surface / repair: the full tessellation (default: only the surface nodes, ~2x faster)'; Group: ''),
    (Flag: '--raster-voxel'; Caption: 'Raster voxel (mm)'; Kind: okFloat; Default: 'size/3';
     Hint: 'remesh / repair / shapes: the spacing of the fields the surfaces (or shapes) are rasterized into'; Group: ''),
    (Flag: '--cdt-fill'; Caption: 'CDT interior spacing'; Kind: okFloat; Default: 'size (0 = none)';
     Hint: 'cdt: the spacing of the interior points (default: the size, else 1.5 x the mean edge)'; Group: ''),
    (Flag: '--opt-rounds'; Caption: 'Optimiser rounds'; Kind: okInt; Default: '3';
     Hint: 'cdt / optimize: rounds of flips, collapses, Steiner points and smoothing'; Group: ''),
    (Flag: '--overlap'; Caption: 'Overlap rule'; Kind: okChoice;
     Default: '(default: nest; shapes: overwrite)|nest|split|max|min|union|cells|overwrite';
     Hint: 'who owns a volume two regions claim -- surfaces that cross (remesh / repair / cdt) or shape objects ' +
       'that overlap: nest: the smaller, split: halfway, max / min: the label, union: one region, cells: each ' +
       'overlap a region of its own (every surface kept), overwrite: the later shape object (MCX)'; Group: ''),
    (Flag: '--auto-labels'; Caption: 'Unlabelled cells'; Kind: okChoice; Default: '(default: cell)|depth';
     Hint: 'surfaces without labels: each enclosed cell a region (outermost, then largest first), or the ' +
       'number of surfaces around it'; Group: ''),
    (Flag: '--size'; Caption: 'Element size (mm)'; Kind: okFloat; Default: '3 x voxel';
     Hint: 'default element size'; Group: 'Sizing'),
    (Flag: '--hmin'; Caption: 'Smallest size (mm)'; Kind: okFloat; Default: 'size/3';
     Hint: 'smallest element size'; Group: ''),
    (Flag: '--hmax'; Caption: 'Largest size (mm)'; Kind: okFloat; Default: 'size';
     Hint: 'largest element size'; Group: ''),
    (Flag: '--lsize'; Caption: 'Per-label size'; Kind: okText; Default: 'L:H,..';
     Hint: 'per-label element size (mm), e.g. 1:4,2:3'; Group: ''),
    (Flag: '--isize'; Caption: 'Interface size'; Kind: okText; Default: 'H|L:H|A:B:H,..';
     Hint: 'element size at interfaces only: every interface (H), every interface of label L ' +
       '(0 = outer surface), or the A|B interface; e.g. 0:2,3:4:1.5'; Group: ''),
    (Flag: '--K'; Caption: 'Elements / radian'; Kind: okFloat; Default: '3';
     Hint: 'elements per radian of curvature'; Group: ''),
    (Flag: '--grad'; Caption: 'Size gradient'; Kind: okFloat; Default: '0.3';
     Hint: 'sizing gradient limit'; Group: ''),
    (Flag: '--thick'; Caption: 'Thin layers (B)'; Kind: okFloat; Default: '0 = off';
     Hint: 'thin layers: h <= local thickness / B'; Group: ''),
    (Flag: '-q'; Caption: 'Max radius-edge'; Kind: okFloat; Default: '2.0';
     Hint: 'max radius-edge ratio (0 = off)'; Group: 'Quality'),
    (Flag: '--opt'; Caption: 'Sliver repair'; Kind: okChoice; Default: '(default: on)|0|1';
     Hint: '3-2/2-3 flips, collapses, Steiner points'; Group: ''),
    (Flag: '--smooth'; Caption: 'ODT passes'; Kind: okInt; Default: '5';
     Hint: 'quality-guarded ODT passes over the interior nodes'; Group: ''),
    (Flag: '--repair'; Caption: 'Repair rounds'; Kind: okInt; Default: '6';
     Hint: 'max restricted-Delaunay repair rounds'; Group: ''),
    (Flag: '--manifold'; Caption: 'Manifold surfaces'; Kind: okBool; Default: '';
     Hint: 'no pinched edges: where two parts of a region touch only along an edge (voxels touching ' +
       'diagonally), one wedge of tets there takes the other label'; Group: ''),
    (Flag: '--nest'; Caption: 'Nesting (outer first)'; Kind: okText; Default: 'L1,L2,.. e.g. 3,4,5';
     Hint: 'labels outermost first (CSF, GM, WM): at a pinch the inner label is joined, and a piece of an ' +
       'outer label cut off inside inner ones merges into them (implies --manifold)'; Group: ''),
    (Flag: '--relax'; Caption: 'Relaxation step'; Kind: okChoice; Default: '(default: fire)|fire|jacobi';
     Hint: 'FIRE (inertial, adaptive time step) or the Jacobi step'; Group: 'Relaxation'),
    (Flag: '--iters'; Caption: 'Relax iterations'; Kind: okInt; Default: '500';
     Hint: 'max relaxation iterations'; Group: ''),
    (Flag: '--thin'; Caption: 'Seed thinning (B)'; Kind: okFloat; Default: '0 = off';
     Hint: 'drop each seed with a kept one of its interface / label closer than B*h (e.g. 0.7)'; Group: ''),
    (Flag: '--sigma'; Caption: 'Smoothing (voxels)'; Kind: okFloat; Default: '1';
     Hint: 'indicator smoothing'; Group: ''),
    (Flag: '--trap'; Caption: 'Boundary trapping'; Kind: okChoice; Default: '(default: smooth)|smooth|voxel';
     Hint: 'smooth sub-voxel interface or exact voxel faces'; Group: ''),
    (Flag: '--thresholds'; Caption: 'Thresholds'; Kind: okText; Default: 'T1,T2,..';
     Hint: 'gray-scale input: label = number of thresholds <= intensity'; Group: 'Gray-scale input'),
    (Flag: '--gray-sigma'; Caption: 'Pre-smoothing'; Kind: okFloat; Default: '0';
     Hint: 'Gaussian pre-smoothing of the gray-scale input (voxels)'; Group: ''),
    (Flag: '--tpm-fields'; Caption: 'Interfaces from probabilities'; Kind: okBool; Default: '';
     Hint: 'the interfaces are smoothed p_a = p_b instead of the argmax labels'; Group: 'Probability maps'),
    (Flag: '--tpm-thresh'; Caption: 'Thresholds'; Kind: okText; Default: 'T|L:T,.. (0.5)';
     Hint: 'per-label threshold: label = argmax(p_l - t_l + 0.5)'; Group: ''),
    (Flag: '--tpm-spm6'; Caption: 'Merge to SPM6 classes'; Kind: okBool; Default: '';
     Hint: 'merge the 18 siamize classes to GM WM CSF Bone Soft'; Group: ''),
    (Flag: '--tpm-exterior'; Caption: 'Exterior channels'; Kind: okText; Default: 'C,.. (auto)';
     Hint: 'exterior channels (0-based)'; Group: ''),
    (Flag: '--tpm-map'; Caption: 'Channel labels'; Kind: okText; Default: 'L0,L1,..';
     Hint: 'label of each channel (0 = exterior; shared = summed)'; Group: ''),
    (Flag: '--tpm-sigma'; Caption: 'Smoothing (voxels)'; Kind: okFloat; Default: '0';
     Hint: 'Gaussian smoothing of the probabilities'; Group: ''),
    (Flag: '--tpm-holes'; Caption: 'Keep exterior pockets'; Kind: okBool; Default: '';
     Hint: 'keep the enclosed exterior pockets (default: filled)'; Group: ''),
    (Flag: '--shape-gap'; Caption: 'Gap closing'; Kind: okFloat; Default: '0.5 (0 = exact)';
     Hint: 'shape (.json) input: facing surfaces closer than about F/2 x size merge (a tangent contact)'; Group: 'Shapes (SDF)'),
    (Flag: '--shape-clip'; Caption: 'Cut to the first object'; Kind: okChoice; Default: '(default: 1)|0|1';
     Hint: 'shape (.json) input: cut every object to the first one''s shape (0: plain painter order)'; Group: ''),
    (Flag: '--gpu'; Caption: 'OpenCL device'; Kind: okFlagArg; Default: 'first GPU';
     Hint: 'relax on an OpenCL device (blank: the first GPU)'; Group: 'Run'),
    (Flag: '--preserve'; Caption: 'Preserve voxel labels'; Kind: okFloat; Default: '0 = off';
     Hint: 'keep each voxel''s own label on top of the smoothed fields by margin M'; Group: ''),
    (Flag: '--jseed'; Caption: 'Junction seeds'; Kind: okFloat; Default: '0.8';
     Hint: 'junction-line seeds, one per C x spacing cell (0 = off)'; Group: ''),
    (Flag: '--no-corners'; Caption: 'No fixed corners'; Kind: okBool; Default: '';
     Hint: 'no fixed nodes where >= 4 labels meet'; Group: ''));

  { --mode per item of the Make choice }
  ModeNames: array[0..5] of string = ('mesh', 'surface', 'remesh', 'repair', 'cdt', 'optimize');

  ClipSteps = 200;

function P3(x, y, z: Single): TI2MPoint;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

{ ------------------------------------------------------------- building --- }

constructor TI2MMainForm.Create(AOwner: TComponent);
const
  Icons: array[0..6] of string = ('open', 'tetmesh', 'run', 'stop', 'saveas', 'save', 'fit');
var
  G: TBitmap;
  C: TPanel;
  k: Integer;
begin
  inherited Create(AOwner);   { the designed form, i2mmain.lfm }
  FShapes := TI2MShapeDoc.Create;
  FShapeSel := -1;
  { the toolbar's icons (i2micons.lrs), in the buttons' ImageIndex order }
  for k := 0 to High(Icons) do
  begin
    G := I2MIcon(Icons[k], ActionIcons.Width);
    if G <> nil then
    begin
      ActionIcons.Add(G, nil);
      G.Free;
    end;
  end;
  G := I2MIcon('reset', I2MIconSize);
  if G <> nil then
  begin
    ResetClipButton.Glyph.Assign(G);
    G.Free;
  end;
  FClip[0] := ClipXFrom;
  FClip[1] := ClipXTo;
  FClip[2] := ClipYFrom;
  FClip[3] := ClipYTo;
  FClip[4] := ClipZFrom;
  FClip[5] := ClipZTo;
  { the titles' glyphs (kept out of the .lfm: plain ASCII there) }
  MeshingChevron.Caption := #$E2#$96#$BE;   { U+25BE, a small down triangle }
  DisplayChevron.Caption := MeshingChevron.Caption;
  ShapesChevron.Caption := MeshingChevron.Caption;
  MeshingClose.Caption := #$C3#$97;         { U+00D7, a multiplication sign }
  DisplayClose.Caption := MeshingClose.Caption;
  ShapesClose.Caption := MeshingClose.Caption;
  for k := 0 to ComponentCount - 1 do
    if (Components[k] is TPanel) and (SectionBody(TPanel(Components[k])) <> nil) then
      PaintHead(TPanel(Components[k]), False);
  BuildMeshingSections;
  OpenSection('Mode');
  OpenSection('Crop box');
  { the designed layout, for Reset layout; then the saved one }
  SetLength(FDefaults, Length(AllCards));
  for k := 0 to High(AllCards) do FDefaults[k] := AllCards[k].BoundsRect;
  LoadLayout;
  { the first field would take the focus and hide its greyed default }
  ActiveControl := MeshingBody;
  FView := TI2MView.Create(ViewHost, I2MGLMode);
  FView.OnLog := @ViewLog;
  EmptyHint.Color := FView.BackgroundColor;
  FEmptyText := EmptyHintText.Caption;
  EmptyHint.BringToFront;   { on the view (its GL window / paint box), under the cards }
  for C in AllCards do C.BringToFront;
  Log('display: ' + FView.Backend);
  FTimer := TTimer.Create(Self);
  FTimer.Enabled := False;
  FTimer.Interval := 100;
  FTimer.OnTimer := @Poll;
  FClipTimer := TTimer.Create(Self);
  FClipTimer.Enabled := False;
  FClipTimer.Interval := 60;
  FClipTimer.OnTimer := @ClipTimer;
  ExeEdit.Text := FindV2mesh;
  DisplayChanged(nil);
  UpdateCommand;
  UpdateButtons;
end;

destructor TI2MMainForm.Destroy;
begin
  if Running then FProc.Terminate(1);
  Application.RemoveAsyncCalls(Self);
  FreeAndNil(FProc);
  FreeAndNil(FView);
  FreeAndNil(FMesh);
  FreeAndNil(FShapes);
  if (FShapeTemp <> '') and FileExists(FShapeTemp) then DeleteFile(FShapeTemp);
  if (FOutFile <> '') and FileExists(FOutFile) then DeleteFile(FOutFile);
  if (FOutFile <> '') and FileExists(ChangeFileExt(FOutFile, '') + '-in' + ExtractFileExt(FOutFile)) then
    DeleteFile(ChangeFileExt(FOutFile, '') + '-in' + ExtractFileExt(FOutFile));
  inherited Destroy;
end;

function TI2MMainForm.Next: Integer;
begin
  Inc(FOrder, 10);
  Result := FOrder;
end;

procedure TI2MMainForm.BuildMeshingSections;
var
  Box: TPanel;   { the current section's body }
  Row: TPanel;
  Lefts: array of TControl;   { the rows' left-hand captions: one width, fitted }
  Bmp: TBitmap;
  w, k: Integer;
  i: Integer;
  L: TLabel;
  E: TEdit;
  C: TCheckBox;
  Cb: TComboBox;
  Items: TStringArray;
  s: string;

  function NewRow: TPanel;
  begin
    Result := TPanel.Create(Self);
    Result.Parent := Box;
    Result.Align := alTop;
    Result.Height := 28;
    Result.BevelOuter := bvNone;
    Result.Top := Next;
  end;

  function RowLabel(ARow: TPanel; const ACaption: string): TLabel;
  begin
    Result := TLabel.Create(Self);
    Result.Parent := ARow;
    Result.Caption := ACaption;
    Result.Align := alLeft;
    Result.Width := 150;
    SetLength(Lefts, Length(Lefts) + 1);
    Lefts[High(Lefts)] := Result;
    Result.AutoSize := False;
    Result.Layout := tlCenter;
    Result.BorderSpacing.Left := 6;
  end;

  procedure Heading(const ACaption: string);
  begin
    Box := SectionBody(SectionHead(ACaption));
    if Box = nil then raise Exception.Create('i2mmain.lfm has no section "' + ACaption + '"');
  end;

begin
  Box := nil;
  { the designed rows' captions share the fitted column }
  Lefts := [ExeLabel, FormatLabel];
  SetLength(FEdits, Length(Options));
  SetLength(FArgEdits, Length(Options));
  for i := 0 to High(Options) do
  begin
    if Options[i].Group <> '' then Heading(Options[i].Group);
    Row := NewRow;
    Row.Hint := Options[i].Flag + ': ' + Options[i].Hint;
    Row.ShowHint := True;
    case Options[i].Kind of
      okBool:
        begin
          C := TCheckBox.Create(Self);
          C.Parent := Row;
          C.Align := alClient;
          C.Caption := Options[i].Caption;
          C.BorderSpacing.Left := 6;
          C.OnChange := @OptionChanged;
          C.Hint := Row.Hint;
          C.ShowHint := True;
          FEdits[i] := C;
        end;
      okChoice:
        begin
          L := RowLabel(Row, Options[i].Caption);
          L.Hint := Row.Hint;
          Cb := TComboBox.Create(Self);
          Cb.Parent := Row;
          Cb.Align := alClient;
          Cb.Style := csDropDownList;
          Items := Options[i].Default.Split('|');
          for s in Items do Cb.Items.Add(s);
          Cb.ItemIndex := 0;
          Cb.BorderSpacing.Right := 4;
          Cb.OnChange := @OptionChanged;
          Cb.Hint := Row.Hint;
          Cb.ShowHint := True;
          FEdits[i] := Cb;
        end;
      okFlagArg:
        begin
          C := TCheckBox.Create(Self);
          C.Parent := Row;
          C.Align := alLeft;
          C.Width := 150;
          SetLength(Lefts, Length(Lefts) + 1);
          Lefts[High(Lefts)] := C;
          C.Caption := Options[i].Caption;
          C.BorderSpacing.Left := 6;
          C.OnChange := @OptionChanged;
          C.Hint := Row.Hint;
          C.ShowHint := True;
          FEdits[i] := C;
          E := TEdit.Create(Self);
          E.Parent := Row;
          E.Align := alClient;
          E.TextHint := Options[i].Default;
          E.BorderSpacing.Right := 4;
          E.OnChange := @OptionChanged;
          E.Hint := Row.Hint;
          E.ShowHint := True;
          FArgEdits[i] := E;
        end;
    else
      begin
        L := RowLabel(Row, Options[i].Caption);
        L.Hint := Row.Hint;
        E := TEdit.Create(Self);
        E.Parent := Row;
        E.Align := alClient;
        E.TextHint := Options[i].Default;
        E.BorderSpacing.Right := 4;
        E.OnChange := @OptionChanged;
        E.Hint := Row.Hint;
        E.ShowHint := True;
        FEdits[i] := E;
      end;
    end;
  end;

  { the captions' column: as wide as the widest caption (a check box: plus its
    box), so none is cut and the fields get the rest of the narrow panel }
  Bmp := TBitmap.Create;
  try
    Bmp.SetSize(4, 4);
    Bmp.Canvas.Font.Assign(Font);
    w := 0;
    for k := 0 to High(Lefts) do
      w := Max(w, Bmp.Canvas.TextWidth(Lefts[k].Caption) + IfThen(Lefts[k] is TCheckBox, MulDiv(26, Screen.PixelsPerInch, 96), 0));
  finally
    Bmp.Free;
  end;
  for k := 0 to High(Lefts) do Lefts[k].Width := w + MulDiv(8, Screen.PixelsPerInch, 96);
end;

{ -------------------------------------------------------------- logging --- }

procedure TI2MMainForm.Log(const AText: string);
begin
  LogMemo.Lines.Add(AText);
  LogMemo.SelStart := Length(LogMemo.Text);
  if FEcho then WriteLn(AText);
end;

procedure TI2MMainForm.ViewLog(Sender: TObject; const AText: string);
begin
  Log(AText);
end;

{ -------------------------------------------------------------- options --- }

function TI2MMainForm.FindV2mesh: string;
var
  Here, BinName, c: string;
  Cands: array of string;
begin
  {$IFDEF WINDOWS}BinName := 'v2mesh.exe';{$ELSE}BinName := 'v2mesh';{$ENDIF}
  Here := ExtractFilePath(ExpandFileName(ParamStr(0)));
  Cands := [Here + BinName, Here + '../build/' + BinName, Here + '../../build/' + BinName,
    Here + '../bin/' + BinName, Here + '../../bin/' + BinName];
  for c in Cands do
    if FileExists(c) then Exit(ExpandFileName(c));
  Result := ExeSearch(BinName, GetEnvironmentVariable('PATH'));
  if Result = '' then Result := BinName;
end;

function TI2MMainForm.Arguments(out AList: TStringList): Boolean;
var
  i: Integer;
  v: string;
  Extra: TStringList;
begin
  AList := TStringList.Create;
  AList.Add('-i');
  if MeshInput then
  begin
    Result := FMeshFile <> '';
    if FMeshFile <> '' then AList.Add(InputMeshFile) else AList.Add('<mesh>');
  end
  else
  begin
    if FShapesOn then
    begin   { the design: its file, or (edited, never saved) a copy in the temporary folder }
      Result := True;
      if FShapesDirty or (FVolFile = '') then
      begin
        if FShapeTemp = '' then
          FShapeTemp := IncludeTrailingPathDelimiter(GetTempDir(False)) + Format('v2m-%d-shapes.json', [GetProcessID]);
        try
          FShapes.SaveToFile(FShapeTemp);
        except
          on E: Exception do Log('could not write ' + FShapeTemp + ': ' + E.Message);
        end;
        AList.Add(FShapeTemp);
      end
      else AList.Add(FVolFile);
    end
    else
    begin
      Result := FVolFile <> '';
      if FVolFile <> '' then AList.Add(FVolFile) else AList.Add('<image>');
    end;
  end;
  if FormatCombo.ItemIndex = 0 then v := '.jmsh' else v := '.bmsh';
  if FOutFile = '' then
    FOutFile := IncludeTrailingPathDelimiter(GetTempDir(False)) +
      Format('v2m-%d', [GetProcessID]) + v
  else
    FOutFile := ChangeFileExt(FOutFile, v);
  AList.Add('-o');
  AList.Add(FOutFile);
  AList.Add('-v');
  for i := 0 to High(Options) do
    case Options[i].Kind of
      okBool:
        if TCheckBox(FEdits[i]).Checked then AList.Add(Options[i].Flag);
      okChoice:
        if TComboBox(FEdits[i]).ItemIndex > 0 then
        begin
          AList.Add(Options[i].Flag);
          if Options[i].Flag = '--mode' then AList.Add(Mode)
          else AList.Add(TComboBox(FEdits[i]).Text);
        end;
      okFlagArg:
        if TCheckBox(FEdits[i]).Checked then
        begin
          AList.Add(Options[i].Flag);
          v := Trim(FArgEdits[i].Text);
          if v <> '' then AList.Add(v);
        end;
    else
      begin
        v := Trim(TEdit(FEdits[i]).Text);
        if v <> '' then
        begin
          AList.Add(Options[i].Flag);
          AList.Add(v);
        end;
      end;
    end;
  if Trim(ExtraEdit.Text) <> '' then
  begin
    Extra := TStringList.Create;
    try
      CommandToList(Trim(ExtraEdit.Text), Extra);
      AList.AddStrings(Extra);
    finally
      Extra.Free;
    end;
  end;
end;

function TI2MMainForm.Mode: string;
var
  i: Integer;
begin
  Result := 'mesh';
  for i := 0 to High(Options) do
    if (Options[i].Flag = '--mode') and (FEdits[i] <> nil) then
      Result := ModeNames[Max(0, TComboBox(FEdits[i]).ItemIndex)];
end;

function TI2MMainForm.MeshInput: Boolean;
begin
  Result := (Mode <> 'mesh') and (Mode <> 'surface');
end;

function TI2MMainForm.InputMeshFile: string;
begin
  { the last result is overwritten by the run: it goes in as a copy }
  if (FMeshFile <> '') and (FOutFile <> '') and (ExpandFileName(FMeshFile) = ExpandFileName(FOutFile)) then
    Result := ChangeFileExt(FOutFile, '') + '-in' + ExtractFileExt(FOutFile)
  else
    Result := FMeshFile;
end;

procedure TI2MMainForm.UpdateCommand;
var
  L: TStringList;
  s, a: string;
begin
  if CmdEdit = nil then Exit;
  Arguments(L);
  try
    s := ExeEdit.Text;
    for a in L do
      if (Pos(' ', a) > 0) or (a = '') then s := s + ' "' + a + '"' else s := s + ' ' + a;
    CmdEdit.Text := s;
  finally
    L.Free;
  end;
end;

procedure TI2MMainForm.OptionChanged(Sender: TObject);
begin
  UpdateCommand;
  UpdateButtons;
  { the argmax view follows v2mesh's channel options }
  if (FVol.Nc > 1) and (ChannelCombo.ItemIndex = 0) and (ArgmaxKey <> FArgmaxKey) then
    ShowChannel;
end;

procedure TI2MMainForm.SetOption(const AFlag, AValue: string);
var
  i: Integer;
begin
  for i := 0 to High(Options) do
    if Options[i].Flag = AFlag then
    begin
      case Options[i].Kind of
        okBool: TCheckBox(FEdits[i]).Checked := True;
        okChoice:
          if AFlag = '--mode' then
            TComboBox(FEdits[i]).ItemIndex := Max(0, AnsiIndexStr(AValue, ModeNames))
          else
            TComboBox(FEdits[i]).ItemIndex := Max(0, TComboBox(FEdits[i]).Items.IndexOf(AValue));
        okFlagArg:
          begin
            TCheckBox(FEdits[i]).Checked := True;
            FArgEdits[i].Text := AValue;
          end;
      else
        TEdit(FEdits[i]).Text := AValue;
      end;
      UpdateCommand;
      UpdateButtons;
      Exit;
    end;
  ExtraEdit.Text := Trim(ExtraEdit.Text + ' ' + AFlag + ' ' + AValue);
end;

procedure TI2MMainForm.BrowseExeClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  try
    D.Title := 'The v2mesh executable';
    D.FileName := ExeEdit.Text;
    if D.Execute then ExeEdit.Text := D.FileName;
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.UpdateButtons;
begin
  if BtnRun = nil then Exit;
  if FView <> nil then
  begin   { over the empty view: what to do (a shape file has no preview) }
    EmptyHint.Visible := not FView.HasContent;
    if FVolFile <> '' then EmptyHintText.Caption := ExtractFileName(FVolFile) + ': nothing to preview -- Run meshes it'
    else EmptyHintText.Caption := FEmptyText;
  end;
  if MeshInput then BtnRun.Enabled := (FMeshFile <> '') and not Running
  else BtnRun.Enabled := ((FVolFile <> '') or FShapesOn) and not Running;
  BtnStop.Enabled := Running;
  BtnSave.Enabled := (FMesh <> nil) and (FMeshFile <> '');
  BtnOpen.Enabled := not Running;
end;

{ -------------------------------------------------------------- display --- }

procedure TI2MMainForm.DisplayChanged(Sender: TObject);
begin
  if (FView = nil) or FUpdating then Exit;
  FView.ShowVolume := ShowVolCheck.Checked;
  FView.ShowMesh := ShowMeshCheck.Checked;
  FView.ShowEdges := ShowEdgesCheck.Checked;
  FView.Colormap := MapCombo.ItemIndex;
  FView.Style := StyleCombo.ItemIndex;
  FView.Opacity := OpacityTrack.Position / 100;
  FView.Threshold := FloorTrack.Position / 100;
  FView.MeshAlpha := MeshAlphaTrack.Position / 100;   { a uniform: no rebuild }
  FView.Redraw;
end;

procedure TI2MMainForm.SetClip(const ALo, AHi: TMcxVec3);
begin
  FUpdating := True;
  try
    FClip[0].Position := Round(ALo.x * ClipSteps);
    FClip[1].Position := Round(AHi.x * ClipSteps);
    FClip[2].Position := Round(ALo.y * ClipSteps);
    FClip[3].Position := Round(AHi.y * ClipSteps);
    FClip[4].Position := Round(ALo.z * ClipSteps);
    FClip[5].Position := Round(AHi.z * ClipSteps);
  finally
    FUpdating := False;
  end;
  ClipChanged(nil);
end;

procedure TI2MMainForm.ClipChanged(Sender: TObject);
var
  i: Integer;
  f: array[0..5] of Single;
begin
  if (FView = nil) or FUpdating then Exit;
  { keep each "from" below its "to" }
  for i := 0 to 2 do
    if FClip[2 * i].Position > FClip[2 * i + 1].Position then
    begin
      FUpdating := True;
      if Sender = FClip[2 * i] then FClip[2 * i + 1].Position := FClip[2 * i].Position
      else FClip[2 * i].Position := FClip[2 * i + 1].Position;
      FUpdating := False;
    end;
  for i := 0 to 5 do f[i] := FClip[i].Position / ClipSteps;
  FView.ClipLo := McxVec3(f[0], f[2], f[4]);
  FView.ClipHi := McxVec3(f[1], f[3], f[5]);
  if Sender = nil then
    FView.ClipChanged
  else
  begin   { the volume follows at once; the mesh once the slider rests }
    FView.Redraw;
    FClipTimer.Enabled := False;
    FClipTimer.Enabled := True;
  end;
end;

procedure TI2MMainForm.ClipTimer(Sender: TObject);
begin
  FClipTimer.Enabled := False;
  FView.ClipChanged;
end;

procedure TI2MMainForm.ResetClipClick(Sender: TObject);
begin
  SetClip(McxVec3(0, 0, 0), McxVec3(1, 1, 1));
end;

{ the displayed size of a voxel: the header's (NIfTI pixdim, JNIfTI VoxelSize),
  so the image and the mesh show in millimetres }
function TI2MMainForm.VoxelSize: TMcxVec3;
begin
  Result := McxVec3(1, 1, 1);
  if FVol.Nx = 0 then Exit;
  if FVol.VoxelSize[0] > 0 then Result.x := FVol.VoxelSize[0];
  if FVol.VoxelSize[1] > 0 then Result.y := FVol.VoxelSize[1];
  if FVol.VoxelSize[2] > 0 then Result.z := FVol.VoxelSize[2];
end;

{ the labels of the mesh and of the image shown (a label volume, or a 4-D
  map's argmax), one list: unticking one hides it in both }
procedure TI2MMainForm.FillLabels;
var
  Seen: array of Boolean;
  t, k: Integer;
  n: string;

  procedure Mark(ATag: Integer);
  begin
    if ATag < 0 then Exit;
    if ATag > High(Seen) then SetLength(Seen, ATag + 1);
    Seen[ATag] := True;
  end;

begin
  Seen := nil;
  if FMesh <> nil then
    for t in FMesh.Labels do Mark(t);
  for t := 1 to High(FVolLabels) do   { 0 is the exterior: never drawn }
    if FVolLabels[t] then Mark(t);
  LabelList.Items.BeginUpdate;
  try
    LabelList.Items.Clear;
    for t := 0 to High(Seen) do
      if Seen[t] then
      begin
        n := 'label ' + IntToStr(t);
        if (t <= High(FLabelNames)) and (FLabelNames[t] <> '') then n := n + ' (' + FLabelNames[t] + ')';
        k := LabelList.Items.AddObject(n, TObject(PtrInt(t)));
        LabelList.Checked[k] := FView.LabelVisible[t];
      end;
  finally
    LabelList.Items.EndUpdate;
  end;
end;

procedure TI2MMainForm.LabelsChanged(Sender: TObject);
var
  k: Integer;
begin
  for k := 0 to LabelList.Items.Count - 1 do
    FView.LabelVisible[PtrInt(LabelList.Items.Objects[k])] := LabelList.Checked[k];
  FView.MeshChanged;
  UploadVolume;
  UpdateStats;
end;

procedure TI2MMainForm.HideLabels(const AList: string);
var
  k, t: Integer;
  it: string;
begin
  for it in AList.Split([','], TStringSplitOptions.ExcludeEmpty) do
    if TryStrToInt(Trim(it), t) then
      for k := 0 to LabelList.Items.Count - 1 do
        if PtrInt(LabelList.Items.Objects[k]) = t then LabelList.Checked[k] := False;
  LabelsChanged(nil);
end;

procedure TI2MMainForm.AllLabelsClick(Sender: TObject);
var
  k: Integer;
begin
  for k := 0 to LabelList.Items.Count - 1 do
    LabelList.Checked[k] := TComponent(Sender).Tag = 0;
  LabelsChanged(nil);
end;

{ the image to the view; a label image with its hidden labels zeroed }
procedure TI2MMainForm.UploadVolume;
var
  Src: PSingle;
  Masked: TI2MSingles;
  nv, i: Int64;
  t: Integer;
  Any: Boolean;
  Gone: array of Boolean;
begin
  if (FVol.Nx = 0) or (FDispPtr = nil) then Exit;
  nv := Int64(FVol.Nx) * FVol.Ny * FVol.Nz;
  Src := FDispPtr;
  Any := False;
  SetLength(Gone, Length(FVolLabels));
  if FDispIsLabel then
    for t := 1 to High(FVolLabels) do
    begin
      Gone[t] := FVolLabels[t] and not FView.LabelVisible[t];
      Any := Any or Gone[t];
    end;
  if Any then
  begin
    SetLength(Masked, nv);
    for i := 0 to nv - 1 do
    begin
      t := Round(Src[i]);
      if (t > 0) and (t <= High(Gone)) and Gone[t] then Masked[i] := 0
      else Masked[i] := Src[i];
    end;
    Src := @Masked[0];
  end;
  FView.SetVolume(Src, FVol.Nx, FVol.Ny, FVol.Nz, FDispLo, FDispHi, VoxelSize);
end;

{ which labels the image shown has (FDispIsLabel only) }
procedure TI2MMainForm.ScanVolumeLabels;
var
  nv, i: Int64;
  t, mx: Integer;
begin
  FVolLabels := nil;
  if not FDispIsLabel or (FDispPtr = nil) then Exit;
  nv := Int64(FVol.Nx) * FVol.Ny * FVol.Nz;
  mx := Round(FDispHi);
  if (mx < 0) or (mx > 65535) then Exit;
  SetLength(FVolLabels, mx + 1);
  for i := 0 to nv - 1 do
  begin
    t := Round(FDispPtr[i]);
    if (t >= 0) and (t <= mx) then FVolLabels[t] := True;
  end;
end;

{ the channel shown: one of a 4-D image, or the argmax }
procedure TI2MMainForm.ShowChannel;
var
  nv, c, nl: Integer;
  Lab: TI2MSingles;
  Map: TI2MIntegers;
  Err, Desc: string;
  lo, hi: Single;
  i: Integer;
begin
  if FVol.Nx = 0 then Exit;
  nv := FVol.Nx * FVol.Ny * FVol.Nz;
  c := ChannelCombo.ItemIndex - 1;
  if (FVol.Nc > 1) and (c < 0) then
  begin   { v2mesh's labels: its exterior channels, maps and thresholds }
    FArgmaxKey := ArgmaxKey;
    if not I2MChannelMap(FVol, OptionText('--tpm-map'), OptionText('--tpm-exterior'), Map, Err) then
    begin
      Log('argmax: ' + Err + '; showing channel indices');
      SetLength(Map, FVol.Nc);
      for i := 0 to FVol.Nc - 1 do Map[i] := i;
    end;
    I2MArgmax(FVol, Map, OptionText('--tpm-thresh'), Lab, nl);
    Desc := '';
    for i := 0 to FVol.Nc - 1 do
    begin
      if Desc <> '' then Desc := Desc + ', ';
      Desc := Desc + ChannelName(i) + ' -> ' + IfThen(Map[i] = 0, 'exterior', IntToStr(Map[i]));
    end;
    Log('argmax labels: ' + Desc);
    FArgmax := Lab;
    FDispPtr := @FArgmax[0];
    FDispLo := 0;
    FDispHi := Max(1, nl - 1);
    FDispIsLabel := True;
    { a label's name: its channels' }
    FLabelNames := nil;
    SetLength(FLabelNames, nl);
    for i := 0 to FVol.Nc - 1 do
      if (Map[i] > 0) and (i <= High(FVol.Names)) and (FVol.Names[i] <> '') then
      begin
        if FLabelNames[Map[i]] <> '' then FLabelNames[Map[i]] := FLabelNames[Map[i]] + '+';
        FLabelNames[Map[i]] := FLabelNames[Map[i]] + FVol.Names[i];
      end;
  end
  else
  begin
    c := Max(0, c);
    lo := FVol.Data[c * nv];
    hi := lo;
    for i := c * nv to (c + 1) * nv - 1 do
    begin
      if FVol.Data[i] < lo then lo := FVol.Data[i];
      if FVol.Data[i] > hi then hi := FVol.Data[i];
    end;
    FArgmax := nil;
    FDispPtr := @FVol.Data[c * nv];
    FDispLo := lo;
    FDispHi := hi;
    FDispIsLabel := (FVol.Nc = 1) and FVol.IsInteger;
    FLabelNames := nil;
  end;
  ScanVolumeLabels;
  FillLabels;
  UploadVolume;
end;

function TI2MMainForm.ChannelName(AChannel: Integer): string;
begin
  Result := IntToStr(AChannel);
  if (AChannel <= High(FVol.Names)) and (FVol.Names[AChannel] <> '') then
    Result := Result + ' (' + FVol.Names[AChannel] + ')';
end;

function TI2MMainForm.OptionText(const AFlag: string): string;
var
  i: Integer;
begin
  Result := '';
  for i := 0 to High(Options) do
    if (Options[i].Flag = AFlag) and (FEdits[i] is TEdit) then
      Exit(Trim(TEdit(FEdits[i]).Text));
end;

function TI2MMainForm.ArgmaxKey: string;
begin
  Result := OptionText('--tpm-map') + '|' + OptionText('--tpm-exterior') + '|' +
    OptionText('--tpm-thresh');
end;

procedure TI2MMainForm.ChannelChanged(Sender: TObject);
begin
  ShowChannel;
end;

{ ---------------------------------------------------------------- files --- }

function TI2MMainForm.LoadImage(const AFileName: string): Boolean;
var
  Err: string;
  c: Integer;
  T0: QWord;
begin
  T0 := GetTickCount64;
  if LowerCase(ExtractFileExt(AFileName)) = '.json' then
  begin   { shape constructs: into the Shapes panel, drawn; v2mesh meshes them }
    if not FShapes.LoadFromFile(AFileName, Err) then
    begin
      Log('could not read ' + AFileName + ': ' + Err);
      StatusBar.SimpleText := 'could not read ' + ExtractFileName(AFileName);
      Exit(False);
    end;
    ShapesInput;
    FVolFile := ExpandFileName(AFileName);
    FShapesDirty := False;
    FShapeSel := IfThen(FShapes.Count > 0, 0, -1);
    FShapeFitted := False;
    RememberDir(AFileName);
    Log(ExtractFileName(AFileName) + Format(': %d shape constructs (MCX Shapes / JMesh Shape*, CSG*); Run meshes them',
      [FShapes.Count]));
    Caption := 'v2m - ' + ExtractFileName(AFileName);
    StatusBar.SimpleText := ExtractFileName(AFileName) + ': shape constructs';
    RefreshShapes;
    ResetBox;
    UpdateCommand;
    Exit(True);
  end;
  { an image: the input now (a design stays in the panel, not drawn) }
  FShapesOn := False;
  FView.ClearShapes;
  Result := I2MLoadVolume(AFileName, FVol, Err);
  if Result then RememberDir(AFileName);
  if not Result then
  begin
    Log('could not read ' + AFileName + ': ' + Err);
    StatusBar.SimpleText := 'could not read ' + ExtractFileName(AFileName);
    Exit;
  end;
  FVolFile := ExpandFileName(AFileName);
  Log(Format('%s: %d x %d x %d%s, %s, %.4g .. %.4g, voxel %.3g x %.3g x %.3g mm (%d ms)',
    [ExtractFileName(AFileName), FVol.Nx, FVol.Ny, FVol.Nz,
     IfThen(FVol.Nc > 1, Format(' x %d channels', [FVol.Nc]), ''),
     IfThen(FVol.IsInteger, 'integer (labels)', 'real'), FVol.Low, FVol.High,
     FVol.VoxelSize[0], FVol.VoxelSize[1], FVol.VoxelSize[2], GetTickCount64 - T0]));
  ChannelCombo.Items.Clear;
  if FVol.Nc > 1 then
  begin
    ChannelCombo.Items.Add('argmax (labels)');
    for c := 0 to FVol.Nc - 1 do ChannelCombo.Items.Add('channel ' + ChannelName(c));
    ChannelCombo.ItemIndex := 0;
    ChannelCombo.Enabled := True;
  end
  else
  begin
    ChannelCombo.Items.Add('(one channel)');
    ChannelCombo.ItemIndex := 0;
    ChannelCombo.Enabled := False;
  end;
  { a label image: the exterior (0) hidden, one colour per label }
  FUpdating := True;
  if FVol.IsInteger or (FVol.Nc > 1) then
  begin
    MapCombo.ItemIndex := 0;
    FloorTrack.Position := 1;
  end
  else
  begin
    MapCombo.ItemIndex := 4;
    FloorTrack.Position := 5;
  end;
  FUpdating := False;
  DisplayChanged(nil);
  if FVol.Oriented then
    FView.Orientation := I2MAxisLetter(FVol.Affine, 0) + I2MAxisLetter(FVol.Affine, 1) + I2MAxisLetter(FVol.Affine, 2)
  else
    FView.Orientation := '';
  ShowChannel;
  { a mesh already open moves into this image's voxels }
  if FMesh <> nil then LoadMesh(FMeshFile, False);
  Caption := 'v2m - ' + ExtractFileName(AFileName);
  StatusBar.SimpleText := Format('%s: %d x %d x %d%s', [ExtractFileName(AFileName),
    FVol.Nx, FVol.Ny, FVol.Nz, IfThen(FVol.Nc > 1, Format(' x %d', [FVol.Nc]), '')]);
  ResetBox;
  UpdateCommand;
  UpdateButtons;
end;

procedure TI2MMainForm.ResetBox;
begin
  SetClip(McxVec3(0, 0, 0), McxVec3(1, 1, 1));
  ResetView;
end;

function TI2MMainForm.LoadMesh(const AFileName: string; AReset: Boolean): Boolean;
var
  M: TI2MMesh;
  T0: QWord;
  First: Boolean;
begin
  T0 := GetTickCount64;
  M := TI2MMesh.Create;
  Result := M.LoadFromFile(AFileName);
  if not Result then
  begin
    Log('could not read ' + AFileName + ': ' + M.Error);
    M.Free;
    Exit;
  end;
  { v2mesh writes world = affine x (i, j, k); the texture has voxel i's
    centre at i + 0.5 }
  if FVol.Nx > 0 then
    with VoxelSize do M.ToDisplay(FVol.Affine, 0.5, P3(x, y, z));
  First := FMesh = nil;
  FView.SetMesh(nil);
  FreeAndNil(FMesh);
  FMesh := M;
  FMeshFile := AFileName;
  if AReset then RememberDir(AFileName);   { (not a run's result, in the temporary folder) }
  { a dense mesh's wireframe is a solid colour at any ordinary zoom }
  if First then
  begin
    ShowOnly('mesh');   { the image drawn over it hides it; one click brings it back }
  end;
  FView.SetMesh(FMesh);
  FView.ShowShapes := False;   { (the mesh of the design replaces its preview) }
  FillLabels;
  UpdateStats;
  Log(Format('%s: %d nodes, %d tets, %d surface triangles (%d ms)',
    [ExtractFileName(AFileName), M.NodeCount, M.ElemCount, M.FaceCount, GetTickCount64 - T0]));
  if M.IsSurface then
    StatusBar.SimpleText := Format('surface: %d nodes, %d triangles', [M.NodeCount, M.FaceCount])
  else
    StatusBar.SimpleText := Format('mesh: %d nodes, %d tets', [M.NodeCount, M.ElemCount]);
  if AReset then ResetBox
  else if First and (FVol.Nx = 0) then FitView;
  UpdateButtons;
end;

procedure TI2MMainForm.OpenClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Open an image';
    D.Filter := 'Images and shapes (*.nii;*.nii.gz;*.jnii;*.bnii;*.json)|*.nii;*.nii.gz;*.gz;*.jnii;*.bnii;*.json|All files|*';
    if D.Execute then LoadImage(D.FileName);
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.MeshClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Open a mesh';
    D.Filter := 'Meshes and surfaces (*.jmsh;*.bmsh;*.off;*.stl)|*.jmsh;*.bmsh;*.off;*.stl|All files|*';
    if D.Execute then LoadMesh(D.FileName);
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.SaveClick(Sender: TObject);
var
  D: TSaveDialog;
  Src, Dst: TFileStream;
begin
  if FMeshFile = '' then Exit;
  D := TSaveDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Save the mesh as';
    D.DefaultExt := Copy(ExtractFileExt(FMeshFile), 2, 8);
    D.Filter := 'Mesh (*' + ExtractFileExt(FMeshFile) + ')|*' + ExtractFileExt(FMeshFile);
    if FVolFile <> '' then
      D.FileName := ChangeFileExt(ChangeFileExt(ExtractFileName(FVolFile), ''), '') + ExtractFileExt(FMeshFile);
    D.Options := D.Options + [ofOverwritePrompt];
    if not D.Execute then Exit;
    if ExpandFileName(D.FileName) = ExpandFileName(FMeshFile) then Exit;
    Src := TFileStream.Create(FMeshFile, fmOpenRead or fmShareDenyWrite);
    try
      Dst := TFileStream.Create(D.FileName, fmCreate);
      try
        Dst.CopyFrom(Src, 0);
      finally
        Dst.Free;
      end;
    finally
      Src.Free;
    end;
    Log('saved ' + D.FileName);
    RememberDir(D.FileName);
  finally
    D.Free;
  end;
end;

{ the default view, from the front (anterior), framed }
procedure TI2MMainForm.ResetView;
begin
  FView.DefaultAngles;
  FitView;
end;

procedure TI2MMainForm.ShowOnly(const AWhat: string);
begin
  ShowVolCheck.Checked := AWhat <> 'mesh';
  ShowMeshCheck.Checked := AWhat <> 'volume';
  DisplayChanged(nil);
end;

procedure TI2MMainForm.ShowPage(AIndex: Integer);
begin
  case AIndex of
    1: OpenSection('Crop box');
    2: OpenSection('Mesh Quality');
  else
    OpenSection('Mode');
  end;
end;

procedure TI2MMainForm.OpenSection(const ACaption: string);
var
  H: TPanel;
begin
  H := SectionHead(ACaption);
  if H = nil then Exit;
  CardOf(H).Visible := True;
  SetCardCollapsed(CardOf(H), False);
  if not SectionBody(H).Visible then OpenHead(H);
  UpdatePanelsMenu;
end;

procedure TI2MMainForm.FitView(AMargin: Boolean);
var
  L, R: Integer;
  C: TPanel;
begin
  { the part of the view the cards leave free: one at the left edge covers up
    to its right side, one at the right edge from its left side }
  L := 0;
  R := 0;
  if AMargin then
    for C in AllCards do
      if C.Visible then
        if C.Left + C.Width div 2 < ViewHost.ClientWidth div 2 then L := Max(L, C.BoundsRect.Right)
        else R := Max(R, ViewHost.ClientWidth - C.Left);
  FView.FitView(L, R);
end;

procedure TI2MMainForm.ShotClick(Sender: TObject);
var
  D: TSaveDialog;
begin
  D := TSaveDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Save the view as';
    D.DefaultExt := 'png';
    D.Filter := 'PNG (*.png)|*.png';
    D.Options := D.Options + [ofOverwritePrompt];
    if D.Execute then
      if SaveImage(D.FileName, ViewHost.Width, ViewHost.Height) then
      begin
        Log('saved ' + D.FileName);
        RememberDir(D.FileName);
      end;
  finally
    D.Free;
  end;
end;

function TI2MMainForm.SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
begin
  Result := FView.SaveImage(AFileName, AWidth, AHeight);
end;

function CopyMesh(const ASrc, ADst: string): Boolean;
var
  Src, Dst: TFileStream;
begin
  Result := False;
  try
    Src := TFileStream.Create(ASrc, fmOpenRead or fmShareDenyWrite);
    try
      Dst := TFileStream.Create(ADst, fmCreate);
      try
        Dst.CopyFrom(Src, 0);
      finally
        Dst.Free;
      end;
    finally
      Src.Free;
    end;
    Result := True;
  except
    on E: Exception do Result := False;
  end;
end;

{ what a dropped / named file is: 1 an image, 2 a mesh, 0 neither }
function FileKind(const AFileName: string): Integer;
var
  n: string;
begin
  n := LowerCase(ExtractFileName(AFileName));
  if n.EndsWith('.jmsh') or n.EndsWith('.bmsh') or n.EndsWith('.off') or n.EndsWith('.stl') then Exit(2);
  if n.EndsWith('.nii') or n.EndsWith('.nii.gz') or n.EndsWith('.jnii') or
     n.EndsWith('.bnii') or n.EndsWith('.json') then Exit(1);
  Result := 0;
end;

procedure TI2MMainForm.FormDropFiles(Sender: TObject; const FileNames: array of string);
var
  f: string;
  k: Integer;
begin
  { images first, so a mesh dropped with its image lands on it }
  for k := 1 to 2 do
    for f in FileNames do
      if FileKind(f) = k then
      begin
        if Running and (k = 1) then
        begin
          Log('v2mesh is running; not opening ' + ExtractFileName(f));
          Continue;
        end;
        if k = 1 then LoadImage(f) else LoadMesh(f);
      end;
  for f in FileNames do
    if FileKind(f) = 0 then
      Log('not an image (.nii .nii.gz .jnii .bnii) or a mesh (.jmsh .bmsh .off .stl): ' + f);
  BringToFront;
end;

{ ---------------------------------------------------------------- running --- }

function TI2MMainForm.Running: Boolean;
begin
  Result := (FProc <> nil) and FProc.Running;
end;

function TI2MMainForm.Busy: Boolean;
begin
  Result := FTimer.Enabled;
end;

procedure TI2MMainForm.Run;
var
  L: TStringList;
begin
  if Running then Exit;
  if not Arguments(L) then
  begin
    L.Free;
    if MeshInput then Log('open a mesh or a surface first (--mode ' + Mode + ')')
    else Log('open an image first');
    Exit;
  end;
  FreeAndNil(FProc);
  if MeshInput and (InputMeshFile <> FMeshFile) then
    if not CopyMesh(FMeshFile, InputMeshFile) then
    begin
      L.Free;
      Log('could not copy ' + FMeshFile + ' to ' + InputMeshFile);
      Exit;
    end;
  if FileExists(FOutFile) then DeleteFile(FOutFile);
  FProc := TProcess.Create(nil);
  FProc.Executable := ExeEdit.Text;
  FProc.Parameters.Assign(L);
  L.Free;
  FProc.Options := [poUsePipes, poStderrToOutPut, poNoConsole];
  FPending := '';
  Log('');
  Log('$ ' + CmdEdit.Text);
  try
    FProc.Execute;
  except
    on E: Exception do
    begin
      Log('could not start ' + ExeEdit.Text + ': ' + E.Message);
      FreeAndNil(FProc);
      UpdateButtons;
      Exit;
    end;
  end;
  FStarted := Now;
  FTimer.Enabled := True;
  StatusBar.SimpleText := 'v2mesh is running...';
  UpdateButtons;
end;

procedure TI2MMainForm.RunClick(Sender: TObject);
begin
  Run;
end;

procedure TI2MMainForm.StopClick(Sender: TObject);
begin
  if Running then
  begin
    FProc.Terminate(1);
    Log('stopped');
  end;
end;

procedure TI2MMainForm.Drain;
var
  Buf: array[0..4095] of Char;
  n, p: Integer;
  Chunk: string;
begin
  if (FProc = nil) or (FProc.Output = nil) then Exit;
  while FProc.Output.NumBytesAvailable > 0 do
  begin
    n := FProc.Output.Read(Buf, Min(SizeOf(Buf), FProc.Output.NumBytesAvailable));
    if n <= 0 then Break;
    SetString(Chunk, PChar(@Buf[0]), n);
    FPending := FPending + Chunk;
  end;
  { whole lines to the log; a carriage return (a progress line) ends one too }
  FPending := StringReplace(FPending, #13#10, #10, [rfReplaceAll]);
  repeat
    p := Pos(#10, FPending);
    if p = 0 then p := Pos(#13, FPending);
    if p = 0 then Break;
    Log(Copy(FPending, 1, p - 1));
    Delete(FPending, 1, p);
  until False;
end;

procedure TI2MMainForm.Poll(Sender: TObject);
begin
  Drain;
  if (FProc <> nil) and not FProc.Running then
  begin
    FTimer.Enabled := False;
    Drain;
    Finished;
  end
  else
    StatusBar.SimpleText := Format('v2mesh is running... %.1f s', [(Now - FStarted) * 86400]);
end;

procedure TI2MMainForm.Finished;
var
  Code: Integer;
begin
  if FPending <> '' then
  begin
    Log(FPending);
    FPending := '';
  end;
  Code := FProc.ExitStatus;
  Log(Format('v2mesh finished in %.1f s, exit code %d', [(Now - FStarted) * 86400, Code]));
  if (Code = 0) and FileExists(FOutFile) then
  begin
    LoadMesh(FOutFile, False);
    ShowOnly('mesh');
  end
  else
    StatusBar.SimpleText := Format('v2mesh failed (exit code %d); see the log', [Code]);
  UpdateButtons;
end;

procedure TI2MMainForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if Running then FProc.Terminate(1);
  SaveLayout;
  CloseAction := caFree;
end;

{ ---------------------------------------------------------------- cards --- }

const
  Snap = 8;
  { v2m.ini's layout: a file of another version (another set of panels, or
    pixels not at 96 dpi) is ignored, and replaced when the window closes }
  LayoutVersion = 3;   { a card dragged this close to the view's edge sticks to it }

function TI2MMainForm.AllCards: TI2MPanels;
begin
  Result := [MeshingCard, DisplayCard, ShapesCard];
end;

{ the card a control is on: its ancestor right under the view host }
function TI2MMainForm.CardOf(AControl: TControl): TPanel;
begin
  Result := nil;
  while (AControl <> nil) and (AControl.Parent <> ViewHost) do AControl := AControl.Parent;
  if AControl is TPanel then Result := TPanel(AControl);
end;

{ a card's body: its client-aligned child (under the title) }
function TI2MMainForm.CardBody(ACard: TPanel): TControl;
var
  k: Integer;
begin
  for k := 0 to ACard.ControlCount - 1 do
    if ACard.Controls[k].Align = alClient then Exit(ACard.Controls[k]);
  Result := nil;
end;

{ collapsed: the title only; its height when open is kept in the card's Tag }
procedure TI2MMainForm.SetCardCollapsed(ACard: TPanel; ACollapsed: Boolean);
var
  B: TControl;
  Chev: TLabel;
  h, bottom: Integer;
begin
  B := CardBody(ACard);
  if (B = nil) or (B.Visible = not ACollapsed) then Exit;
  Chev := TLabel(FindComponent(ACard.Name.Replace('Card', 'Chevron')));
  bottom := ACard.Top + ACard.Height;
  if ACollapsed then
  begin
    ACard.Tag := ACard.Height;
    B.Visible := False;
    h := ACard.Height - ACard.ClientHeight + 2 * ACard.BorderWidth +
      TControl(FindComponent(ACard.Name.Replace('Card', 'Title'))).Height;
    if Chev <> nil then Chev.Caption := #$E2#$96#$B8;   { U+25B8, a small right triangle }
  end
  else
  begin
    B.Visible := True;
    h := ACard.Tag;
    if h <= 0 then h := ACard.Height;
    if Chev <> nil then Chev.Caption := #$E2#$96#$BE;
  end;
  if akBottom in ACard.Anchors then ACard.SetBounds(ACard.Left, bottom - h, ACard.Width, h)   { its bottom stays }
  else ACard.Height := h;
  KeepInView(ACard);
  CheckStats;
end;

{ a card inside the view; ASnap (a drag): against its edge when near it }
procedure TI2MMainForm.KeepInView(ACard: TPanel; ASnap: Boolean);
var
  L, T, W, H, s: Integer;
begin
  W := ViewHost.ClientWidth;
  H := ViewHost.ClientHeight;
  if (ACard.Width > W) or (ACard.Height > H) then   { (no larger than the view) }
    ACard.SetBounds(ACard.Left, ACard.Top, Min(ACard.Width, W), Min(ACard.Height, H));
  L := ACard.Left;
  T := ACard.Top;
  if ASnap then s := Snap else s := 0;
  if L + ACard.Width > W - s then L := W - ACard.Width;
  if T + ACard.Height > H - s then T := H - ACard.Height;
  if L < s then L := 0;
  if T < s then T := 0;
  if (L <> ACard.Left) or (T <> ACard.Top) then ACard.SetBounds(L, T, ACard.Width, ACard.Height);
end;

procedure TI2MMainForm.CardTitleMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  if Button <> mbLeft then Exit;
  FDragCard := CardOf(TControl(Sender));
  if FDragCard = nil then Exit;
  FDragFrom := Mouse.CursorPos;
  FDragOrigin := Point(FDragCard.Left, FDragCard.Top);
  FDragCard.BringToFront;
end;

procedure TI2MMainForm.CardTitleMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
var
  P: TPoint;
begin
  if (FDragCard = nil) or not (ssLeft in Shift) then Exit;
  P := Mouse.CursorPos;
  FDragCard.SetBounds(FDragOrigin.X + P.X - FDragFrom.X, FDragOrigin.Y + P.Y - FDragFrom.Y,
    FDragCard.Width, FDragCard.Height);
  KeepInView(FDragCard, True);
end;

procedure TI2MMainForm.CardTitleMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var
  P: TPoint;
  C: TPanel;
begin
  C := FDragCard;
  FDragCard := nil;
  if C = nil then Exit;
  P := Mouse.CursorPos;
  if (Abs(P.X - FDragFrom.X) < 4) and (Abs(P.Y - FDragFrom.Y) < 4) then   { a click, not a drag }
    SetCardCollapsed(C, CardBody(C).Visible)
  else if FView <> nil then FView.Redraw;   { what the card uncovered }
end;

procedure TI2MMainForm.CardCollapseClick(Sender: TObject);
var
  C: TPanel;
begin
  C := CardOf(TControl(Sender));
  if C <> nil then SetCardCollapsed(C, CardBody(C).Visible);
end;

const
  EdgeLeft = 1;
  EdgeRight = 2;
  EdgeTop = 4;
  EdgeBottom = 8;
  MinCardW = 220;
  MinCardH = 120;

{ which edges of card C the point (X, Y) (its own) is on: its frame (BorderWidth
  and the border line), a little more at a corner }
function CardEdges(C: TPanel; X, Y: Integer): Integer;
var
  g: Integer;
begin
  g := C.BorderWidth + 2;
  Result := 0;
  if X < g then Result := Result or EdgeLeft;
  if X >= C.ClientWidth - g then Result := Result or EdgeRight;
  if Y < g then Result := Result or EdgeTop;
  if Y >= C.ClientHeight - g then Result := Result or EdgeBottom;
end;

procedure TI2MMainForm.CardEdgeMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var
  C: TPanel;
begin
  if Button <> mbLeft then Exit;
  C := TPanel(Sender);
  FSizeEdges := CardEdges(C, X, Y);
  if (CardBody(C) <> nil) and not CardBody(C).Visible then   { collapsed: sideways only }
    FSizeEdges := FSizeEdges and (EdgeLeft or EdgeRight);
  if FSizeEdges = 0 then Exit;
  FSizeCard := C;
  FSizeFrom := Mouse.CursorPos;
  FSizeOrigin := C.BoundsRect;
  C.BringToFront;
end;

procedure TI2MMainForm.CardEdgeMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
var
  C: TPanel;
  e, dx, dy: Integer;
  R: TRect;
begin
  C := TPanel(Sender);
  if (FSizeCard = nil) or not (ssLeft in Shift) then
  begin   { the pointer: the edge it would pull }
    e := CardEdges(C, X, Y);
    if (CardBody(C) <> nil) and not CardBody(C).Visible then e := e and (EdgeLeft or EdgeRight);
    case e of
      EdgeLeft, EdgeRight: C.Cursor := crSizeWE;
      EdgeTop, EdgeBottom: C.Cursor := crSizeNS;
      EdgeLeft or EdgeTop, EdgeRight or EdgeBottom: C.Cursor := crSizeNWSE;
      EdgeRight or EdgeTop, EdgeLeft or EdgeBottom: C.Cursor := crSizeNESW;
    else
      C.Cursor := crDefault;
    end;
    Exit;
  end;
  dx := Mouse.CursorPos.X - FSizeFrom.X;
  dy := Mouse.CursorPos.Y - FSizeFrom.Y;
  R := FSizeOrigin;
  if FSizeEdges and EdgeLeft <> 0 then R.Left := Min(Max(0, R.Left + dx), R.Right - MinCardW);
  if FSizeEdges and EdgeRight <> 0 then R.Right := Max(Min(ViewHost.ClientWidth, R.Right + dx), R.Left + MinCardW);
  if FSizeEdges and EdgeTop <> 0 then R.Top := Min(Max(0, R.Top + dy), R.Bottom - MinCardH);
  if FSizeEdges and EdgeBottom <> 0 then R.Bottom := Max(Min(ViewHost.ClientHeight, R.Bottom + dy), R.Top + MinCardH);
  FSizeCard.BoundsRect := R;
end;

procedure TI2MMainForm.CardEdgeMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  if FSizeCard = nil then Exit;
  FSizeCard := nil;
  if FView <> nil then FView.Redraw;   { what the card uncovered }
end;

procedure TI2MMainForm.CardCloseClick(Sender: TObject);
var
  C: TPanel;
begin
  C := CardOf(TControl(Sender));
  if C = nil then Exit;
  C.Visible := False;
  UpdatePanelsMenu;
  if FView <> nil then FView.Redraw;
end;

{ ------------------------------------------------------------- sections --- }

{ SectXxxHead's body is SectXxxBody (nil: not a section title) }
function TI2MMainForm.SectionBody(AHead: TPanel): TPanel;
var
  C: TComponent;
begin
  Result := nil;
  if (AHead = nil) or not AHead.Name.EndsWith('Head') then Exit;
  C := FindComponent(Copy(AHead.Name, 1, Length(AHead.Name) - 4) + 'Body');
  if C is TPanel then Result := TPanel(C);
end;

{ a section title's text, without the glyph PaintHead puts before it }
function HeadText(AHead: TPanel): string;
begin
  Result := TrimLeft(AHead.Caption);
  if (Result <> '') and (Ord(Result[1]) >= $80) then Result := Copy(Result, Pos(' ', Result) + 1, MaxInt);
end;

{ the section titled ACaption }
function TI2MMainForm.SectionHead(const ACaption: string): TPanel;
var
  k: Integer;
begin
  for k := 0 to ComponentCount - 1 do
    if (Components[k] is TPanel) and (SectionBody(TPanel(Components[k])) <> nil) and
       SameText(HeadText(TPanel(Components[k])), ACaption) then
      Exit(TPanel(Components[k]));
  Result := nil;
end;

{ its glyph, and its colours -- the theme's (light or dark): open or under the
  pointer, its selection colours; else its button colours }
procedure TI2MMainForm.PaintHead(AHead: TPanel; AHot: Boolean);
var
  s: string;
  B: TPanel;
begin
  B := SectionBody(AHead);
  if B = nil then Exit;
  s := HeadText(AHead);
  if B.Visible then AHead.Caption := '  ' + #$E2#$96#$BE + ' ' + s
  else AHead.Caption := '  ' + #$E2#$96#$B8 + ' ' + s;
  if B.Visible or AHot then
  begin
    AHead.Color := clHighlight;
    AHead.Font.Color := clHighlightText;
  end
  else
  begin
    AHead.Color := clBtnFace;
    AHead.Font.Color := clBtnText;
  end;
end;

{ opens AHead's section, closing the others of its card (one open at a time);
  an open one's title closes it }
procedure TI2MMainForm.OpenHead(AHead: TPanel);
var
  k: Integer;
  H, B: TPanel;
  Opening: Boolean;
begin
  Opening := not SectionBody(AHead).Visible;
  AHead.Parent.DisableAlign;
  try
    for k := 0 to AHead.Parent.ControlCount - 1 do
      if AHead.Parent.Controls[k] is TPanel then
      begin
        H := TPanel(AHead.Parent.Controls[k]);
        B := SectionBody(H);
        if B = nil then Continue;
        { a hidden alTop panel keeps its Top, and is put back by it: just past its
          own title's (at the title's bottom it would tie with the next title) }
        if Opening and (H = AHead) then B.Top := H.Top + 1;
        B.Visible := Opening and (H = AHead);
        PaintHead(H, False);
      end;
  finally
    AHead.Parent.EnableAlign;
  end;
  if Opening and (AHead.Parent is TScrollBox) then TScrollBox(AHead.Parent).ScrollInView(AHead);
  CheckStats;
end;

procedure TI2MMainForm.SectionHeadClick(Sender: TObject);
begin
  OpenHead(TPanel(Sender));
  PaintHead(TPanel(Sender), True);
end;

procedure TI2MMainForm.SectionHeadEnter(Sender: TObject);
begin
  PaintHead(TPanel(Sender), True);
end;

procedure TI2MMainForm.SectionHeadLeave(Sender: TObject);
begin
  PaintHead(TPanel(Sender), False);
end;

{ ------------------------------------------------------ the Panels menu --- }

procedure TI2MMainForm.UpdatePanelsMenu;
begin
  MenuMeshing.Checked := MeshingCard.Visible;
  MenuDisplay.Checked := DisplayCard.Visible;
  MenuShapes.Checked := ShapesCard.Visible;
end;

procedure TI2MMainForm.ViewMenuClick(Sender: TObject);
var
  C: TPanel;
  k: Integer;
begin
  case TComponent(Sender).Tag of
    10:
      begin
        FitView;
        Exit;
      end;
    11:
      begin
        ResetView;
        Exit;
      end;
    1: C := MeshingCard;
    2: C := DisplayCard;
    3: C := ShapesCard;
  else
    begin   { Reset layout: the designed places, all shown and open }
      for k := 0 to High(AllCards) do
      begin
        C := AllCards[k];
        C.BoundsRect := FDefaults[k];
        C.Tag := 0;
        C.Visible := (C <> ShapesCard) or FShapesOn;
        if CardBody(C) <> nil then CardBody(C).Visible := True;
      end;
      MeshingChevron.Caption := #$E2#$96#$BE;
      DisplayChevron.Caption := #$E2#$96#$BE;
      ShapesChevron.Caption := #$E2#$96#$BE;
      KeepCardsInView(0);
      UpdatePanelsMenu;
      Exit;
    end;
  end;
  C.Visible := not C.Visible;
  if C.Visible then
  begin
    C.BringToFront;
    KeepInView(C);
    CheckStats;
  end;
  UpdatePanelsMenu;
end;

procedure TI2MMainForm.BtnViewClick(Sender: TObject);
var
  P: TPoint;
begin
  UpdatePanelsMenu;
  P := BtnView.ClientToScreen(Point(0, BtnView.Height));
  ViewMenu.PopUp(P.X, P.Y);
end;

{ the view resized: the cards kept in it -- after this event (GTK2 may resize
  during a paint, and refuses the cards' invalidation then) }
procedure TI2MMainForm.ViewHostResize(Sender: TObject);
begin
  if (ViewHost = nil) or (DisplayCard = nil) or FKeepQueued then Exit;   { (while the form loads) }
  FKeepQueued := True;
  Application.QueueAsyncCall(@KeepCardsInView, 0);
end;

procedure TI2MMainForm.KeepCardsInView(Data: PtrInt);
var
  C: TPanel;
begin
  FKeepQueued := False;
  for C in AllCards do KeepInView(C);
end;

{ ---------------------------------------------------- the mesh's statistics --- }

const
  HistBins = 40;
  FineBins = 1000;   { (for the percentiles) }

{ the mesh or its shown labels changed: the statistics again -- now if the
  Mesh Quality section is on screen, else when it is next shown (CheckStats) }
procedure TI2MMainForm.UpdateStats;
begin
  FStatsDirty := True;
  CheckStats;
end;

procedure TI2MMainForm.CheckStats;
begin
  if FStatsDirty and (SectStatsBody <> nil) and SectStatsBody.IsVisible then ComputeStats;
end;

procedure TI2MMainForm.ComputeStats;
var
  Hidden: array of Boolean;
  Q, V: TI2MValues;
  FineQ, FineV: array of Integer;
  n, k, t, bad: Integer;
  qmin, qsum, vmin, vmax, vsum, lo, hi: Double;
  Kind, SizeName, Unit_: string;

  { the value of fine bin histogram H (over lo..hi, FineBins bins) at fraction p of n }
  function Pct(const H: array of Integer; p, lo_, hi_: Double): Double;
  var
    i, c: Integer;
  begin
    c := 0;
    for i := 0 to High(H) do
    begin
      Inc(c, H[i]);
      if c >= p * n then Exit(lo_ + (i + 0.5) * (hi_ - lo_) / FineBins);
    end;
    Result := hi_;
  end;

begin
  FStatsDirty := False;
  SetLength(FQualHist, HistBins);
  SetLength(FSizeHist, HistBins);
  FillChar(FQualHist[0], HistBins * SizeOf(Integer), 0);
  FillChar(FSizeHist[0], HistBins * SizeOf(Integer), 0);
  QualityHist.Invalidate;
  SizeHist.Invalidate;
  if (FMesh = nil) or (FMesh.NodeCount = 0) then
  begin
    StatsText.Caption := '(no mesh)';
    Exit;
  end;
  SetLength(Hidden, FMesh.MaxTag + 1);
  for t := 0 to High(Hidden) do Hidden[t] := not FView.LabelVisible[t];
  FMesh.ElementStats(Hidden, Q, V);
  n := Length(Q);
  if FMesh.IsSurface then
  begin
    Kind := 'triangles';
    SizeName := 'area';
    Unit_ := 'mm' + #$C2#$B2;   { mm^2 }
  end
  else
  begin
    Kind := 'tets';
    SizeName := 'volume';
    Unit_ := 'mm' + #$C2#$B3;   { mm^3 }
  end;
  SizeCaption.Caption := UpperCase(SizeName[1]) + Copy(SizeName, 2, MaxInt) + ' (' + Unit_ + ', log scale)';
  if n = 0 then
  begin
    StatsText.Caption := Format('no %s shown (the labels ticked in Display)', [Kind]);
    Exit;
  end;
  { quality: 0..1 }
  SetLength(FineQ, FineBins);
  qmin := 1e30;
  qsum := 0;
  bad := 0;
  for k := 0 to n - 1 do
  begin
    t := Min(FineBins - 1, Max(0, Trunc(Q[k] * FineBins)));
    Inc(FineQ[t]);
    Inc(FQualHist[t * HistBins div FineBins]);
    qmin := Min(qmin, Q[k]);
    qsum := qsum + Q[k];
    if Q[k] < 0.1 then Inc(bad);
  end;
  { size: log10, over the positive ones' range }
  vmin := 1e30;
  vmax := 0;
  vsum := 0;
  for k := 0 to n - 1 do
  begin
    vsum := vsum + V[k];
    if V[k] > 0 then
    begin
      vmin := Min(vmin, V[k]);
      vmax := Max(vmax, V[k]);
    end;
  end;
  if vmax <= 0 then
  begin
    vmin := 1;
    vmax := 1;
  end;
  lo := Log10(vmin);
  hi := Log10(vmax);
  if hi - lo < 1e-6 then
  begin
    lo := lo - 0.5;
    hi := hi + 0.5;
  end;
  FSizeLo := lo;
  FSizeHi := hi;
  SetLength(FineV, FineBins);
  for k := 0 to n - 1 do
  begin
    if V[k] > 0 then t := Trunc((Log10(V[k]) - lo) / (hi - lo) * FineBins) else t := 0;
    t := Min(FineBins - 1, Max(0, t));
    Inc(FineV[t]);
    Inc(FSizeHist[t * HistBins div FineBins]);
  end;
  StatsText.Caption :=
    Format('%d %s shown (of %d)', [n, Kind, IfThen(FMesh.IsSurface, FMesh.FaceCount, FMesh.ElemCount)]) + LineEnding +
    Format('%d nodes', [FMesh.NodeCount]) + LineEnding +
    Format('quality: min %.3f, 5%% %.3f', [qmin, Pct(FineQ, 0.05, 0, 1)]) + LineEnding +
    Format('  median %.3f, mean %.3f', [Pct(FineQ, 0.5, 0, 1), qsum / n]) + LineEnding +
    Format('  below 0.1: %d (%.2f%%)', [bad, 100 * bad / n]) + LineEnding +
    Format('%s: min %.4g, median %.4g', [SizeName, vmin, Power(10, Pct(FineV, 0.5, lo, hi))]) + LineEnding +
    Format('  max %.4g, total %.6g %s', [vmax, vsum, Unit_]);
end;

{ a histogram: FQualHist (Tag 0) or FSizeHist (1), in the theme's colours }
procedure TI2MMainForm.HistPaint(Sender: TObject);
var
  PB: TPaintBox;
  C: TCanvas;
  H: array of Integer;
  k, m, x0, x1, y0, plotH, th: Integer;
  lo, hi: string;
begin
  PB := TPaintBox(Sender);
  C := PB.Canvas;
  C.Brush.Style := bsSolid;
  C.Brush.Color := clWindow;
  C.Pen.Color := clBtnShadow;
  C.Rectangle(0, 0, PB.Width, PB.Height);
  C.Font.Height := -11;
  C.Font.Color := clWindowText;
  if PB.Tag = 0 then H := FQualHist else H := FSizeHist;
  m := 0;
  for k := 0 to High(H) do m := Max(m, H[k]);
  th := C.TextHeight('0');
  plotH := PB.Height - th - 6;
  if m = 0 then
  begin
    C.Brush.Style := bsClear;
    C.TextOut(6, (PB.Height - th) div 2, '(nothing to show)');
    Exit;
  end;
  { the bars }
  C.Brush.Color := clHighlight;
  C.Pen.Color := clHighlight;
  for k := 0 to High(H) do
    if H[k] > 0 then
    begin
      x0 := 1 + k * (PB.Width - 2) div Length(H);
      x1 := 1 + (k + 1) * (PB.Width - 2) div Length(H) - 1;
      y0 := 2 + plotH - Max(1, Round(H[k] / m * (plotH - 2)));
      C.Rectangle(x0, y0, Max(x0 + 1, x1), 2 + plotH);
    end;
  { the ticks: the ends, and the peak count }
  C.Brush.Style := bsClear;
  if PB.Tag = 0 then
  begin
    lo := '0';
    hi := '1';
    C.TextOut((PB.Width - C.TextWidth('0.5')) div 2, PB.Height - th - 2, '0.5');
  end
  else
  begin
    lo := Format('%.3g', [Power(10, FSizeLo)]);
    hi := Format('%.3g', [Power(10, FSizeHi)]);
  end;
  C.TextOut(4, PB.Height - th - 2, lo);
  C.TextOut(PB.Width - C.TextWidth(hi) - 4, PB.Height - th - 2, hi);
  C.TextOut(4, 3, Format('peak %d', [m]));
end;

{ -------------------------------------------------------- the Shapes panel --- }

{ the design becomes the input: no image, no mesh; the panel shown }
procedure TI2MMainForm.ShapesInput;
begin
  if not FShapesOn then
  begin
    FView.ClearVolume;
    FVol := Default(TI2MVolume);
    FView.SetMesh(nil);
    FreeAndNil(FMesh);
    FMeshFile := '';
    FillLabels;
    UpdateStats;
    FShapeFitted := False;
  end;
  FShapesOn := True;
  ShapesCard.Visible := True;
  ShapesCard.BringToFront;
  KeepInView(ShapesCard);
  UpdatePanelsMenu;
end;

{ the list, the selected construct's fields, the preview }
procedure TI2MMainForm.RefreshShapes;
var
  i: Integer;
  s: string;
  B: TJSONData;
  N: TTreeNode;
begin
  ShapeTree.Items.BeginUpdate;
  try
    ShapeTree.Items.Clear;
    for i := 0 to FShapes.Count - 1 do
    begin
      s := Format('%d. %s', [i + 1, FShapes.Key(i)]);
      B := FShapes.Body(i);
      if (B is TJSONObject) and (TJSONObject(B).Find('Tag') is TJSONNumber) then
        s := s + Format('  (Tag %d)', [TJSONObject(B).Integers['Tag']]);
      ShapeTree.Items.Add(nil, s);
    end;
  finally
    ShapeTree.Items.EndUpdate;
  end;
  if FShapeSel >= FShapes.Count then FShapeSel := FShapes.Count - 1;
  if FShapeSel >= 0 then
  begin
    N := ShapeTree.Items[FShapeSel];
    ShapeTree.OnSelectionChanged := nil;   { (no second refresh) }
    N.Selected := True;
    ShapeTree.OnSelectionChanged := @ShapeTreeChange;
  end;
  FillShapeFields;
  PreviewShapes;
end;

{ the selected construct's members, one row each, as JSON (a Layers row: "Layer i") }
procedure TI2MMainForm.FillShapeFields;
var
  B: TJSONData;
  i: Integer;
begin
  ShapeFields.Strings.BeginUpdate;
  try
    ShapeFields.Strings.Clear;
    if (FShapeSel < 0) or (FShapeSel >= FShapes.Count) then Exit;
    B := FShapes.Body(FShapeSel);
    if B is TJSONObject then
      for i := 0 to B.Count - 1 do
        ShapeFields.Strings.Add(TJSONObject(B).Names[i] + '=' + B.Items[i].AsJSON)
    else if B is TJSONArray then
      for i := 0 to B.Count - 1 do
        ShapeFields.Strings.Add(Format('Layer %d=%s', [i + 1, B.Items[i].AsJSON]))
    else
      ShapeFields.Strings.Add('value=' + B.AsJSON);
  finally
    ShapeFields.Strings.EndUpdate;
  end;
end;

procedure TI2MMainForm.PreviewShapes;
var
  S: TI2MShapeScene;
begin
  S := FShapes.Build(FShapeSel);
  FView.SetShapes(S);
  FView.ShowShapes := True;
  if S.Error <> '' then ShapeHint.Caption := 'cannot draw ' + S.Error
  else ShapeHint.Caption := Format('%d constructs; later ones overwrite earlier ones, all cut to the first',
    [FShapes.Count]);
  if not FShapeFitted then
  begin
    FShapeFitted := True;
    ResetView;
  end;
  UpdateButtons;
end;

{ an edit: the design is the input, unsaved; drawn again }
procedure TI2MMainForm.ShapesChanged;
begin
  ShapesInput;
  FShapesDirty := True;
  if FVolFile <> '' then Caption := 'v2m - ' + ExtractFileName(FVolFile) + ' (edited)'
  else Caption := 'v2m - (a new design)';
  RefreshShapes;
  UpdateCommand;
end;

procedure TI2MMainForm.ShapeAddBtnClick(Sender: TObject);
var
  P: TPoint;
begin
  P := ShapeAddBtn.ClientToScreen(Point(0, ShapeAddBtn.Height));
  ShapeAddMenu.PopUp(P.X, P.Y);
end;

procedure TI2MMainForm.ShapeAddClick(Sender: TObject);
var
  S: TI2MShapeScene;
  k: string;
  B: TJSONData;
begin
  k := TMenuItem(Sender).Hint;   { (the key: a caption may gain an accelerator's '&') }
  if not FShapesOn then
  begin   { (no design yet: a new, empty one, this its first construct) }
    FShapes.Clear(False);
    FVolFile := '';
    FShapeSel := -1;
  end;
  S := FShapes.Build(-1);
  if FShapes.Count = 0 then
  begin   { (nothing to place it in yet: MCX Studio's 60-voxel domain) }
    S.Lo.x := 0; S.Lo.y := 0; S.Lo.z := 0;
    S.Hi.x := 60; S.Hi.y := 60; S.Hi.z := 60;
  end;
  try
    B := GetJSON(I2MDefaultShape(k, S.Lo, S.Hi, FShapes.MaxTag + 1));
  except
    on E: Exception do
    begin
      ShapeHint.Caption := 'could not add ' + k + ': ' + E.Message;
      Exit;
    end;
  end;
  FShapes.Add(k, B);
  FShapeSel := FShapes.Count - 1;
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeDelClick(Sender: TObject);
begin
  if (FShapeSel < 0) or (FShapeSel >= FShapes.Count) then Exit;
  FShapes.Delete(FShapeSel);
  if FShapeSel >= FShapes.Count then FShapeSel := FShapes.Count - 1;
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeUpClick(Sender: TObject);
begin
  if FShapeSel <= 0 then Exit;
  FShapes.Move(FShapeSel, FShapeSel - 1);
  Dec(FShapeSel);
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeDownClick(Sender: TObject);
begin
  if (FShapeSel < 0) or (FShapeSel >= FShapes.Count - 1) then Exit;
  FShapes.Move(FShapeSel, FShapeSel + 1);
  Inc(FShapeSel);
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeNewClick(Sender: TObject);
begin
  if FShapesOn and FShapesDirty and (MessageDlg('New design', 'Discard the changes to the current design?',
     mtConfirmation, [mbYes, mbNo], 0) <> mrYes) then Exit;
  FShapes.Clear;
  FVolFile := '';
  FShapeSel := 0;
  FShapeFitted := False;
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeOpenClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Open shape constructs';
    D.Filter := 'Shape constructs (*.json)|*.json|All files|*';
    if D.Execute then LoadImage(D.FileName);
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.ShapeSaveClick(Sender: TObject);
var
  D: TSaveDialog;
begin
  D := TSaveDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Save the design as';
    D.DefaultExt := 'json';
    D.Filter := 'Shape constructs (*.json)|*.json';
    D.Options := D.Options + [ofOverwritePrompt];
    if FVolFile <> '' then D.FileName := ExtractFileName(FVolFile);
    if not D.Execute then Exit;
    try
      FShapes.SaveToFile(D.FileName);
    except
      on E: Exception do
      begin
        Log('could not save ' + D.FileName + ': ' + E.Message);
        Exit;
      end;
    end;
    FVolFile := ExpandFileName(D.FileName);
    FShapesDirty := False;
    RememberDir(D.FileName);
    Caption := 'v2m - ' + ExtractFileName(D.FileName);
    Log('saved ' + D.FileName);
    UpdateCommand;
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.ShapeTreeChange(Sender: TObject);
begin
  if ShapeTree.Selected = nil then Exit;
  FShapeSel := ShapeTree.Selected.Index;
  FillShapeFields;
  PreviewShapes;
end;

{ a value edited: parsed as JSON (else taken as a string), into the construct }
procedure TI2MMainForm.ShapeFieldValidate(Sender: TObject; ACol, ARow: Integer; const OldValue: string;
  var NewValue: string);
var
  B, D: TJSONData;
  Name_: string;
  k: Integer;
begin
  if (NewValue = OldValue) or (FShapeSel < 0) or (FShapeSel >= FShapes.Count) or (ARow < 1) then Exit;
  try
    D := GetJSON(NewValue);
  except
    D := TJSONString.Create(NewValue);   { (bare text: a string) }
  end;
  B := FShapes.Body(FShapeSel);
  Name_ := ShapeFields.Keys[ARow];
  if B is TJSONObject then
  begin
    k := TJSONObject(B).IndexOfName(Name_);
    if k >= 0 then TJSONObject(B).Items[k] := D else TJSONObject(B).Add(Name_, D);
  end
  else if (B is TJSONArray) and (ARow - 1 < B.Count) then
    TJSONArray(B).Items[ARow - 1] := D
  else
  begin
    D.Free;
    Exit;
  end;
  NewValue := D.AsJSON;
  FShapesDirty := True;
  ShapesChanged;
end;

{ ------------------------------------------------------- the last folder --- }

{ a file dialog opens in the folder of the last file opened or saved }
procedure TI2MMainForm.UseLastDir(D: TFileDialog);
begin
  if (FLastDir <> '') and DirectoryExists(FLastDir) then D.InitialDir := FLastDir;
end;

procedure TI2MMainForm.RememberDir(const AFileName: string);
begin
  FLastDir := ExtractFileDir(ExpandFileName(AFileName));
end;

{ ------------------------------------------------------ the saved layout --- }

function TI2MMainForm.LayoutFile: string;
begin
  Result := IncludeTrailingPathDelimiter(GetAppConfigDir(False)) + 'v2m.ini';
end;

procedure TI2MMainForm.LoadLayout;
var
  Ini: TIniFile;
  C: TPanel;
  R: TRect;
begin
  if not FileExists(LayoutFile) then Exit;
  Ini := TIniFile.Create(LayoutFile);
  try
    FLastDir := Ini.ReadString('Files', 'LastDir', '');
    if Ini.ReadInteger('Layout', 'Version', 1) <> LayoutVersion then Exit;
    for C in AllCards do
    begin
      if not Ini.SectionExists(C.Name) then Continue;
      { (kept at 96 dpi: the same layout on any screen) }
      R := C.BoundsRect;
      R.Left := Scale96ToScreen(Ini.ReadInteger(C.Name, 'Left', ScaleScreenTo96(R.Left)));
      R.Top := Scale96ToScreen(Ini.ReadInteger(C.Name, 'Top', ScaleScreenTo96(R.Top)));
      R.Right := R.Left + Scale96ToScreen(Ini.ReadInteger(C.Name, 'Width', ScaleScreenTo96(C.Width)));
      R.Bottom := R.Top + Scale96ToScreen(Ini.ReadInteger(C.Name, 'Height', ScaleScreenTo96(C.Height)));
      C.BoundsRect := R;
      C.Visible := Ini.ReadBool(C.Name, 'Visible', True);
      if Ini.ReadBool(C.Name, 'Collapsed', False) then
      begin
        C.Height := Scale96ToScreen(Ini.ReadInteger(C.Name, 'OpenHeight', ScaleScreenTo96(C.Height)));
        SetCardCollapsed(C, True);
      end;
      KeepInView(C);
    end;
  finally
    Ini.Free;
  end;
  UpdatePanelsMenu;
end;

procedure TI2MMainForm.SaveLayout;
var
  Ini: TIniFile;
  C: TPanel;
  Collapsed: Boolean;
begin
  try
    ForceDirectories(ExtractFilePath(LayoutFile));
    Ini := TIniFile.Create(LayoutFile);
    try
      Ini.EraseSection('ActionsCard');   { (of layout 1) }
      Ini.EraseSection('LogCard');
      Ini.WriteInteger('Layout', 'Version', LayoutVersion);
      if FLastDir <> '' then Ini.WriteString('Files', 'LastDir', FLastDir);
      for C in AllCards do
      begin
        Collapsed := (CardBody(C) <> nil) and not CardBody(C).Visible;
        Ini.WriteInteger(C.Name, 'Left', ScaleScreenTo96(C.Left));
        Ini.WriteInteger(C.Name, 'Top', ScaleScreenTo96(C.Top));
        Ini.WriteInteger(C.Name, 'Width', ScaleScreenTo96(C.Width));
        Ini.WriteInteger(C.Name, 'Height', ScaleScreenTo96(C.Height));
        Ini.WriteBool(C.Name, 'Visible', C.Visible);
        Ini.WriteBool(C.Name, 'Collapsed', Collapsed);
        if Collapsed then Ini.WriteInteger(C.Name, 'OpenHeight', ScaleScreenTo96(C.Tag));
      end;
    finally
      Ini.Free;
    end;
  except
    on E: Exception do Log('could not save the layout: ' + E.Message);   { (a read-only home) }
  end;
end;

end.
