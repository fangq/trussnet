{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mmain -- the window: open a volume, set v2mesh's options, run it, look at
  the image and the mesh together, cropped and translucent.

  The window is a designed form (i2mmain.lfm): the main menu (every panel's
  commands, by the same handlers as their buttons; Help > About: i2mabout) and the toolbar on top, the
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
  i2mmesh, i2mview, i2micons, i2mshapes, i2mabout, IntfGraphics, FPImage;

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
  TI2MControls = array of TControl;
  TI2MEdits = array of TEdit;

  TI2MMainForm = class(TForm)
  published
    { the designed form (i2mmain.lfm) }
    ActionBar, ShapeBar: TToolBar;
    BtnOpen, ActionDiv1, BtnRun, BtnStop, ActionDiv2, BtnSave, BtnShot, ActionDiv3,
    BtnView, ShapeAddBtn, ShapeDiv0, ShapeUpBtn, ShapeDownBtn, ShapeDiv1, ShapeNewBtn,
    ShapeOpenBtn, ShapeSaveBtn, ShapeDiv2, ShapeDelBtn: TToolButton;
    StatusBar: TStatusBar;
    LogPanel, ViewHost, MeshingCard, MeshingTitle, SectPathHead, SectPathBody, ExeRow, FormatRow,
    SectModeHead, SectModeBody, SectSizingHead, SectSizingBody, SectQualityHead, SectQualityBody,
    SectRelaxHead, SectRelaxBody, SectGrayHead, SectGrayBody, SectTpmHead, SectTpmBody,
    SectShapesHead, SectShapesBody, SectRunHead, SectRunBody, SectOtherHead, SectOtherBody,
    DisplayCard, DisplayTitle, SectCropHead, SectCropBody, SectLabelsHead, SectLabelsBody,
    LabelButtons, SectVolumeHead, SectVolumeBody, SectMeshHead, SectMeshBody, SectStatsHead,
    SectStatsBody, ShapesCard, ShapesTitle, ShapesBody, EmptyHint: TPanel;
    CmdEdit, ExeEdit, ExtraEdit, PictThreshEdit: TEdit;
    LogMemo: TMemo;
    LogSplitter, ShapeSplit: TSplitter;
    MeshingChevron, MeshingClose, MeshingCaption, ExeLabel, FormatLabel, DisplayChevron,
    DisplayClose, DisplayCaption, ClipLabel0, ClipLabel1, ClipLabel2, ClipLabel3, ClipLabel4,
    ClipLabel5, OrientLabel, ChannelLabel, PictLabel, PictThreshLabel, MapLabel, StyleLabel, OpacityLabel, FloorLabel, MeshAlphaLabel,
    StatsText, QualityCaption, SizeCaption, ShapesChevron, ShapesClose, ShapesCaption, ShapeHint,
    EmptyHintText: TLabel;
    MeshingPin, DisplayPin, ShapesPin: TShape;
    MeshingBody, DisplayBody: TScrollBox;
    ExeBrowse: TButton;
    FormatCombo, OrientCombo, ChannelCombo, PictCombo, MapCombo, StyleCombo: TComboBox;
    ClipXFrom, ClipXTo, ClipYFrom, ClipYTo, ClipZFrom, ClipZTo, OpacityTrack, FloorTrack,
    MeshAlphaTrack: TTrackBar;
    ResetClipButton, ShowAllButton, HideAllButton, MergeButton: TBitBtn;
    LabelList: TCheckListBox;
    ShowVolCheck, ShowMeshCheck, ShowEdgesCheck, InnerOnlyCheck: TCheckBox;
    QualityHist, SizeHist: TPaintBox;
    ShapeTree: TTreeView;
    ShapeFields: TValueListEditor;
    ActionIcons, OpenMenuIcons: TImageList;
    OpenMenu, ViewMenu, ShapeAddMenu: TPopupMenu;
    OpenImageItem, OpenMeshItem, OpenCadItem, MenuFit, MenuResetView, MenuSep1, MenuMeshing, MenuDisplay, MenuShapes, MenuSep2, MenuReset,
    ShapeAddMCX, ShapeAdd_Grid, ShapeAdd_Box, ShapeAdd_Subgrid, ShapeAdd_Sphere, ShapeAdd_Cylinder,
    ShapeAdd_XSlabs, ShapeAdd_YSlabs, ShapeAdd_ZSlabs, ShapeAdd_XLayers, ShapeAdd_YLayers,
    ShapeAdd_ZLayers, ShapeAddJMesh, ShapeAdd_ShapeBox3, ShapeAdd_ShapeSphere,
    ShapeAdd_ShapeCylinder, ShapeAdd_ShapeCone, ShapeAdd_ShapeConeFrustum, ShapeAdd_ShapeEllipsoid,
    ShapeAdd_ShapeTorus, ShapeAdd_ShapeSphereShell, ShapeAdd_ShapeSphereSegment,
    ShapeAdd_ShapePlane3, ShapeAddCSG, ShapeAdd_CSGUnion, ShapeAdd_CSGIntersect,
    ShapeAdd_CSGSubtract: TMenuItem;
    MainMenu: TMainMenu;
    { the brain2mesh (gpu_brain2mesh) and siamize panels }
    B2MCard, B2MTitle, SectB2MPathHead, SectB2MPathBody, B2MExeRow, B2MOutRow, B2MRunRow, SectB2MSurfHead,
    SectB2MSurfBody, SectB2MShellsHead, SectB2MShellsBody, SectB2MTetHead, SectB2MTetBody, SectB2MDevHead,
    SectB2MDevBody, SectB2MOtherHead, SectB2MOtherBody, SiamCard, SiamTitle, SectSiamPathHead, SectSiamPathBody,
    SiamExeRow, SiamOutRow, SiamRunRow, SectSiamSegHead, SectSiamSegBody, SectSiamCompHead, SectSiamCompBody,
    SectSiamResHead, SectSiamResBody, SectSiamOtherHead, SectSiamOtherBody: TPanel;
    B2MChevron, B2MClose, B2MCaption, B2MExeLabel, B2MOutLabel, B2MInLabel, SiamChevron, SiamClose, SiamCaption,
    SiamExeLabel, SiamOutLabel, SiamInLabel: TLabel;
    B2MPin, SiamPin: TShape;
    B2MBody, SiamBody: TScrollBox;
    B2MExeBrowse, B2MOutBrowse, B2MRunBtn, B2MStopBtn, SiamExeBrowse, SiamOutBrowse, SiamRunBtn, SiamStopBtn: TButton;
    B2MExeEdit, B2MOutEdit, B2MCmdEdit, B2MExtraEdit, SiamExeEdit, SiamOutEdit, SiamCmdEdit, SiamExtraEdit: TEdit;
    MenuB2M, MenuSiam, MainPanB2M, MainPanSiam, MainRunB2M, MainRunSiam: TMenuItem;
    MainFile, MainOpenAny, MainOpenVolume, MainOpenMesh, MainOpenCad, MainSep1, MainSaveMesh, MainSaveShot, MainSep2,
    MainExit, MainShapes, MainShapeNew, MainShapeOpen, MainShapeSave, MainSep3, MainShapeAdd, MainAddMCX,
    MainAdd_Grid, MainAdd_Box, MainAdd_Subgrid, MainAdd_Sphere, MainAdd_Cylinder, MainAdd_XSlabs, MainAdd_YSlabs,
    MainAdd_ZSlabs, MainAdd_XLayers, MainAdd_YLayers, MainAdd_ZLayers, MainAddJMesh, MainAdd_ShapeBox3,
    MainAdd_ShapeSphere, MainAdd_ShapeCylinder, MainAdd_ShapeCone, MainAdd_ShapeConeFrustum, MainAdd_ShapeEllipsoid,
    MainAdd_ShapeTorus, MainAdd_ShapeSphereShell, MainAdd_ShapeSphereSegment, MainAdd_ShapePlane3, MainAddCSG,
    MainAdd_CSGUnion, MainAdd_CSGIntersect, MainAdd_CSGSubtract, MainShapeDel, MainShapeUp, MainShapeDown, MainSep4,
    MainShapePanel, MainMesh, MainRun, MainStop, MainSep5, MainMode, MainMode_mesh, MainMode_surface, MainMode_remesh,
    MainMode_repair, MainMode_cdt, MainMode_optimize, MainSettings, MainSect0, MainSect1, MainSect2, MainSect3,
    MainSect4, MainSect5, MainSect6, MainSect7, MainSect8, MainSect9, MainSep6, MainMeshPanel, MainView, MainFit,
    MainResetView, MainSep7, MainShowVol, MainShowMesh, MainShowEdges, MainResetClip, MainDispSettings, MainDSect0,
    MainDSect1, MainDSect2, MainDSect3, MainDSect4, MainSep8, MainPanMeshing, MainPanDisplay, MainPanShapes,
    MainPanReset, MainHelp, MainHelpV2m, MainHelpV2mesh, MainHelpIssues, MainSep9, MainAbout, MainRecent: TMenuItem;
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormShow(Sender: TObject);
    procedure PictChanged(Sender: TObject);
    procedure MergeLabelsClick(Sender: TObject);
    procedure RecentClick(Sender: TObject);
    procedure RecentClearClick(Sender: TObject);
    procedure FormDropFiles(Sender: TObject; const FileNames: array of string);
    procedure OptionChanged(Sender: TObject);
    procedure DisplayChanged(Sender: TObject);
    procedure ClipChanged(Sender: TObject);
    procedure LabelsChanged(Sender: TObject);
    procedure InnerOnlyChanged(Sender: TObject);
    procedure ChannelChanged(Sender: TObject);
    procedure OrientChanged(Sender: TObject);
    procedure OpenClick(Sender: TObject);
    procedure OpenAnyClick(Sender: TObject);
    procedure OpenCadClick(Sender: TObject);
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
    procedure CardPinMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure CardEdgeMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure CardEdgeMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure CardEdgeMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure SectionHeadClick(Sender: TObject);
    procedure SectionHeadEnter(Sender: TObject);
    procedure SectionHeadLeave(Sender: TObject);
    procedure ViewMenuClick(Sender: TObject);
    procedure BtnViewClick(Sender: TObject);
    { the brain2mesh and siamize panels }
    procedure ToolBrowseClick(Sender: TObject);
    procedure ToolChanged(Sender: TObject);
    procedure ToolRunClick(Sender: TObject);
    { the main menu }
    procedure MainMenuOpen(Sender: TObject);
    procedure MainExitClick(Sender: TObject);
    procedure MainModeClick(Sender: TObject);
    procedure MainSectionClick(Sender: TObject);
    procedure MainDisplayClick(Sender: TObject);
    procedure MainHelpClick(Sender: TObject);
    procedure MainAboutClick(Sender: TObject);
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
    FEdits: TI2MControls;   { per option: TEdit / TCheckBox / TComboBox }
    FArgEdits: TI2MEdits;   { okFlagArg: the argument }
    { the brain2mesh / siamize panels' option rows (B2MOptions, SiamOptions) }
    FB2MEdits, FSiamEdits: TI2MControls;
    FB2MArgEdits, FSiamArgEdits: TI2MEdits;
    { the job running (or last run): 0 v2mesh, 1 brain2mesh, 2 siamize; its name and output }
    FJob: Integer;
    FJobName, FJobOut: string;
    FDesignedVisible: array of Boolean;   { the cards' designed visibility (Reset the panels) }
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
    { each card's own place (where it was put: loaded, dragged to, reset), which
      a collapse goes back to -- an expansion may move it to fit the view }
    FHome: array of TPoint;
    { auto-hide: an unpinned card collapses to its title when the pointer has
      been off it a moment, and opens when the pointer is back on it }
    FPinned: array of Boolean;
    FAway: array of QWord;   { when the pointer left it (0: on it) }
    FDefHost: TPoint;        { the view's size when FDefaults were taken (the designed layout) }
    FOver: array of QWord;   { when the pointer came onto it, collapsed (0: not on it) }
    FHoverTimer: TTimer;
    FEmptyText: string;   { the empty view's hint, as designed }
    FKeepQueued: Boolean; { the cards to be kept in the resized view (queued) }
    FLastDir: string;     { the folder of the last file opened or saved (kept in v2m.ini) }
    FGrayMap: Integer;    { the colour map of an intensity volume (MapCombo index; 0 jet; kept in v2m.ini) }
    FPicFile: string;     { the input, when a picture (FVolFile: its one-slice NIfTI copy) }
    FRecent: TStringList; { the files last opened from a dialog or dropped, newest first (kept in v2m.ini) }
    { the shape constructs being designed (the Shapes panel); FShapesOn: they
      are Run's input -- written to FShapeTemp first when edited or unsaved }
    FShapes: TI2MShapeDoc;
    FShapesOn, FShapesDirty: Boolean;
    FShapeSel: Integer;       { the selected node's top-level construct (-1: none) }
    FSelPath: TI2MIntegers;   { the selected node: its top-level index, then its operand indices }
    FNodePaths: array of TI2MIntegers;   { each tree node's path (its Data: the index here) }
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
    FFileOrient: string;   { the image's axis letters by its header ('': none) }
    FMeshSrc: string;   { the CAD / PLC file the mesh shown was read from ('' : the mesh itself) }
    FPair: Boolean;
    FCentred: Boolean;  { (placed on its monitor: once) }
    FShapeGuessed: Boolean;   { the last preview's domain a placeholder (nothing drawn yet) }     { a volume and a mesh dropped together: both kept (the mesh on the volume) }
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
    procedure BuildOptionRows(const AOpts: array of TI2MOption; var AEdits: TI2MControls; var AArgEdits: TI2MEdits;
      const ALefts: array of TControl; AOnChange: TNotifyEvent);
    function ToolArgs(ATool: Integer; out AList: TStringList): Boolean;
    function ToolOut(ATool: Integer): string;
    procedure UpdateToolCommands;
    function FindTool(const ABin, AEnv: string): string;
    procedure RunJob(ATool: Integer; const AExe: string; AArgs: TStringList; const AOut, ACmd: string);
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
    function ChosenMode: string;
    function MeshInput: Boolean;
    procedure ApplyOrientation;
    procedure UpdateFloorLabel;
    procedure SyncMainMenu;
    function InputMeshFile: string;
    { the cards }
    function AllCards: TI2MPanels;
    function CardOf(AControl: TControl): TPanel;
    function CardBody(ACard: TPanel): TControl;
    procedure SetCardCollapsed(ACard: TPanel; ACollapsed: Boolean);
    procedure KeepInView(ACard: TPanel; ASnap: Boolean = False);
    function SectionBody(AHead: TPanel): TPanel;
    function SectionHead(const ACaption: string; ACard: TPanel = nil): TPanel;
    procedure OpenHead(AHead: TPanel);
    procedure PaintHead(AHead: TPanel; AHot: Boolean);
    procedure UpdatePanelsMenu;
    procedure UpdateStats;
    procedure KeepCardsInView(Data: PtrInt);
    procedure ApplyLayout(Data: PtrInt);
    function DefaultRect(k: Integer): TRect;
    function CardIndex(ACard: TPanel): Integer;
    procedure SetHome(ACard: TPanel);
    procedure HoverTick(Sender: TObject);
    procedure PaintPin(ACard: TPanel);
    function Pinned(ACard: TPanel): Boolean;
    procedure UseLastDir(D: TFileDialog);
    procedure ShapesInput;
    procedure RefreshShapes;
    procedure SetSelPath(const APath: TI2MIntegers);
    function ShapeAt(const APath: TI2MIntegers; out AKey: string; out ABody: TJSONData;
      out AParent: TJSONArray; out AIdx: Integer): Boolean;
    procedure FillShapeFields;
    procedure PreviewShapes;
    procedure ShapesChanged;
    procedure ShapeMove(ADelta: Integer);
    procedure RememberDir(const AFileName: string);
    procedure AddRecent(const AFileName: string);
    procedure SaveRecent;
    procedure BuildRecentMenu;
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
    { a CAD model (.step .stp) or a TetGen PLC (.poly .smesh): read by v2mesh
      (--mode convert) into a surface shown as a mesh; the mesh modes then read
      the file itself }
    function LoadCad(const AFileName: string): Boolean;
    { a new volume replaces the mesh, a new mesh the volume (not when dropped together) }
    procedure ClearMesh;
    procedure ClearVolumeData;
    { any file, by its suffix: an image, a mesh, shapes, a CAD model or a PLC }
    function OpenAny(const AFileName: string): Boolean;
    procedure ResetBox;
    { -q 2 --size 3 ...: for the command line and scripted runs }
    procedure SetOption(const AFlag, AValue: string);
    procedure MergeLabels(const AList: string);
    procedure SetClip(const ALo, AHi: TMcxVec3);
    { Display > Volume > Orientation: letters (PSL), or 'file' for the header's }
    procedure SetOrientationText(const AText: string);
    { brain2mesh (1) / siamize (2): run it on the volume open; its panel's other
      arguments; its panel shown }
    procedure RunTool(ATool: Integer);
    procedure SetToolArgs(ATool: Integer; const AArgs: string);
    procedure ShowToolPanel(ATool: Integer);
    function SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
    procedure Run;
    function Running: Boolean;
    function Busy: Boolean;   { running, or its output not yet taken in }
    { AMargin: framed in what the cards leave free (not for a screenshot,
      which has no cards) }
    procedure FitView;
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
    { a volume and a mesh opened together (the command line): both kept }
    property Pair: Boolean read FPair write FPair;
  end;

var
  I2MMainForm: TI2MMainForm;
  { the view's GL: auto / glx / egl / soft (see TI2MView.Create) }
  I2MGLMode: string = 'auto';

implementation

{$R *.lfm}

const
  Options: array[0..48] of TI2MOption = (
    (Flag: '--mode'; Caption: 'Make'; Kind: okChoice;
     Default: 'mesh: tets of the volume|surface: the volume''s surfaces|remesh: tets of the mesh''s surfaces|' +
       'repair: clean surfaces of the mesh|cdt: tets keeping the mesh''s surfaces|optimize: better tets of the mesh';
     Hint: 'what to make of what: the volume (mesh, surface) or the mesh shown (remesh, repair, cdt, optimize; ' +
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
    (Flag: '--maxvol'; Caption: 'Max tet volume (mm3)'; Kind: okFloat; Default: '0 = off';
     Hint: 'cdt / optimize: the largest tet volume (TetGen -a): bigger tets are refined, the surfaces kept'; Group: 'Quality'),
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
    (Flag: '--surf-smooth'; Caption: 'Surface smoothing passes'; Kind: okInt; Default: '0 = off';
     Hint: 'volume-preserving smoothing of the region surfaces after the tessellation (iso2mesh smoothsurf): N passes'; Group: ''),
    (Flag: '--surf-smooth-method'; Caption: 'Surface smoothing'; Kind: okChoice; Default: '(default: laplacianhc)|laplacianhc|lowpass|laplacian';
     Hint: 'laplacianhc: Vollmer''s HC (volume-preserving); lowpass: Taubin; laplacian: plain (shrinks)'; Group: ''),
    (Flag: '--surf-smooth-alpha'; Caption: 'Smoothing alpha'; Kind: okFloat; Default: '0.5';
     Hint: 'the step (HC: the pull back to the original nodes; smaller: smoother)'; Group: ''),
    (Flag: '--surf-smooth-beta'; Caption: 'Smoothing beta (HC)'; Kind: okFloat; Default: '0.5';
     Hint: 'HC''s correction weight'; Group: ''),
    (Flag: '--thresholds'; Caption: 'Thresholds'; Kind: okText; Default: 'T1,T2,..';
     Hint: 'gray-scale input: label = number of thresholds <= intensity'; Group: 'Gray-scale input'),
    (Flag: '--gray-sigma'; Caption: 'Pre-smoothing'; Kind: okFloat; Default: '0';
     Hint: 'Gaussian pre-smoothing of the gray-scale input (voxels)'; Group: ''),
    (Flag: '--tpm-fields'; Caption: 'Interfaces from probabilities'; Kind: okBool; Default: '';
     Hint: 'the interfaces are smoothed p_a = p_b instead of the argmax labels'; Group: 'Probability maps'),
    (Flag: '--tpm-thresh'; Caption: 'Thresholds'; Kind: okText; Default: 'T|L:T,.. (0.5)';
     Hint: 'per-label threshold: label = argmax(p_l - t_l + 0.5)'; Group: ''),
    (Flag: '--tpm-pair'; Caption: 'Pair thresholds'; Kind: okText; Default: 'e.g. 1:13:0.3';
     Hint: 'A:B:T,..: label A''s threshold against label B only (e.g. gray matter vs dura), its other interfaces unchanged'; Group: ''),
    (Flag: '--tpm-gap'; Caption: 'Minimum gaps'; Kind: okText; Default: 'e.g. 1:3:13:0.5';
     Hint: 'A:B:C[+C..]:D,..: a label-B layer at least D mm thick between labels A and C (e.g. CSF between gray matter and dura)'; Group: ''),
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

  { the brain2mesh (gpu_brain2mesh) panel's options: -i the volume open, -o the Output }
  B2MOptions: array[0..18] of TI2MOption = (
    (Flag: '-n'; Caption: 'Relaxation passes'; Kind: okInt; Default: '20';
     Hint: 'SurfaceNets relaxation iterations (0 = off)'; Group: 'Surfaces (SurfaceNets)'),
    (Flag: '-m'; Caption: '18 -> SPM merge'; Kind: okChoice; Default: '(default: spm)|spm|none';
     Hint: 'merge SIAM''s 18 classes into SPM''s 6 tissues (none: keep the 18 -- the csfv shell needs it)'; Group: ''),
    (Flag: '--no-air'; Caption: 'No air cavities'; Kind: okBool; Default: '';
     Hint: 'do not add the enclosed air cavities as a shell'; Group: ''),
    (Flag: '-S'; Caption: 'Shells'; Kind: okText; Default: 'e.g. scalp,skull,csf,gm,wm';
     Hint: 'closed tissue shells, outer to inner, instead of the multi-material surface; name:mm sets one''s remesh edge (with CGAL)'; Group: 'Tissue Shells'),
    (Flag: '--cgal'; Caption: 'CGAL clean-up'; Kind: okBool; Default: '';
     Hint: 'make each shell watertight, simplify it and resolve the crossings (CGAL)'; Group: ''),
    (Flag: '--density'; Caption: 'Keep ratio'; Kind: okFloat; Default: '(0, 1)';
     Hint: 'surface simplification: the fraction of the faces kept (with CGAL)'; Group: ''),
    (Flag: '--smooth'; Caption: 'Smoothing passes'; Kind: okInt; Default: '0';
     Hint: 'Taubin smoothing iterations per shell (with CGAL)'; Group: ''),
    (Flag: '--gpu-tet'; Caption: 'Tets (GPU refiner)'; Kind: okBool; Default: '';
     Hint: 'a tetrahedral mesh: the exact CDT of the surfaces, refined on the GPU (MeshElem)'; Group: 'Tetrahedra'),
    (Flag: '--tet'; Caption: 'Tets (CGAL Mesh_3)'; Kind: okBool; Default: '';
     Hint: 'a tetrahedral mesh by CGAL''s Mesh_3'; Group: ''),
    (Flag: '--radbound'; Caption: 'Cell size (mm)'; Kind: okFloat; Default: '';
     Hint: 'iso2mesh radbound: the surface triangles'' circumradius / the cell size (per tissue: -S name:mm)'; Group: ''),
    (Flag: '--reratio'; Caption: 'Max radius-edge'; Kind: okFloat; Default: '2.0';
     Hint: 'the tets'' radius-edge bound (TetGen -q; 0 = off)'; Group: ''),
    (Flag: '-a'; Caption: 'Max tet volume'; Kind: okText; Default: 'e.g. 100, or gm:2,wm:2';
     Hint: 'the largest tet volume (mm^3, TetGen -a): one number, or per tissue'; Group: ''),
    (Flag: '--lattice'; Caption: 'Interior seeds'; Kind: okChoice; Default: '(default: none)|none|bcc|fcc|hex';
     Hint: 'pre-seed the interior with a regular lattice (spacing from the cell size)'; Group: ''),
    (Flag: '--mindihedral'; Caption: 'Min dihedral (deg)'; Kind: okFloat; Default: '0 = off';
     Hint: 'also refine tets with a smaller dihedral angle (~10 removes the slivers near the surface)'; Group: ''),
    (Flag: '--cluster'; Caption: 'Surface edge (mm)'; Kind: okFloat; Default: '1.25 x cell size';
     Hint: 'the edge the multi-material surface is remeshed to before the CDT'; Group: ''),
    (Flag: '--maxiters'; Caption: 'Max refine rounds'; Kind: okInt; Default: '100';
     Hint: 'a cap on the GPU refinement rounds'; Group: ''),
    (Flag: '--no-surf-refine'; Caption: 'Keep the surface'; Kind: okBool; Default: '';
     Hint: 'refine the volume only (no nodes inserted on the surfaces)'; Group: ''),
    (Flag: '--gpu'; Caption: 'GPU SurfaceNets'; Kind: okBool; Default: '';
     Hint: 'run SurfaceNets on the GPU (OpenCL)'; Group: 'Device'),
    (Flag: '--gpuid'; Caption: 'OpenCL device'; Kind: okInt; Default: 'first GPU';
     Hint: 'the OpenCL device, 1-based (brain2mesh --cl-info lists them)'; Group: ''));

  { the siamize panel's options: -i the volume open (a T1 MRI), -o the Output }
  SiamOptions: array[0..11] of TI2MOption = (
    (Flag: '-M'; Caption: 'Models'; Kind: okChoice; Default: '(default: fold 0)|0|0,1,2,3,4|0,1,2';
     Hint: 'the network folds averaged: more is better and slower (~540 MB each, downloaded on first use)'; Group: 'Segmentation'),
    (Flag: '-C'; Caption: 'Classes'; Kind: okChoice; Default: '(default: 18, SIAM)|18|spm';
     Hint: '18 SIAM classes, or SPM''s 6 tissues (GM, WM, CSF, bone, soft, air)'; Group: ''),
    (Flag: '--tpm'; Caption: 'Probability map (4-D)'; Kind: okBool; Default: '';
     Hint: 'a 4-D tissue probability map instead of the labels (large: ~3 GB raw)'; Group: ''),
    (Flag: '--tpm-t'; Caption: 'TPM temperature'; Kind: okFloat; Default: '1.0';
     Hint: 'softmax temperature of the probability map (> 1: softer)'; Group: ''),
    (Flag: '-c'; Caption: 'Compute'; Kind: okChoice; Default: '(default: auto)|auto|cpu|opencl|vulkan|metal';
     Hint: 'where the network runs (auto: OpenCL when built with it, else the CPU)'; Group: 'Compute'),
    (Flag: '-G'; Caption: 'GPU'; Kind: okText; Default: '0 = auto';
     Hint: 'the GPU: 1-based over the OpenCL devices (siamize -L lists them)'; Group: ''),
    (Flag: '-t'; Caption: 'CPU threads'; Kind: okInt; Default: 'auto';
     Hint: 'CPU worker threads (the CPU backend)'; Group: ''),
    (Flag: '--mnn-fp16'; Caption: 'Half precision (fp16)'; Kind: okBool; Default: '';
     Hint: 'run the GPU in fp16: faster, a little less accurate'; Group: ''),
    (Flag: '--lowmem'; Caption: 'Low memory'; Kind: okBool; Default: '';
     Hint: 'the low-memory preset (for hosts short of RAM or VRAM)'; Group: ''),
    (Flag: '-u'; Caption: 'Spacing (mm)'; Kind: okFloat; Default: '0.75';
     Hint: 'the isotropic spacing the network runs at'; Group: 'Resolution'),
    (Flag: '-P'; Caption: 'Patch (ZxYxX)'; Kind: okText; Default: '256x256x192';
     Hint: 'the sliding window (smaller: less memory)'; Group: ''),
    (Flag: '--upsample'; Caption: 'Keep the 0.75 mm grid'; Kind: okBool; Default: '';
     Hint: 'save at the inference resolution (super-resolved) instead of the input''s grid'; Group: ''));

  { --mode per item of the Make choice }
  ModeNames: array[0..5] of string = ('mesh', 'surface', 'remesh', 'repair', 'cdt', 'optimize');

  ClipSteps = 200;

function P3(x, y, z: Single): TI2MPoint;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

{ A picture through the LCL's TPicture (png bmp jpeg gif tiff pnm xpm ico ...):
  its pixels, $RRGGBB, row 0 at the top (alpha ignored, as the FPC readers do) }
function ReadPictureLCL(const AFileName: string; out W, H: Integer; out ARGB: TI2MIntegers;
  out AError: string): Boolean;
var
  Pic: TPicture;
  Img: TLazIntfImage;
  i, j: Integer;
  c: TFPColor;
begin
  Result := False;
  W := 0;
  H := 0;
  ARGB := nil;
  AError := '';
  Pic := TPicture.Create;
  try
    try
      Pic.LoadFromFile(AFileName);
      if not (Pic.Graphic is TRasterImage) then
      begin
        AError := 'not a raster picture';
        Exit;
      end;
      Img := TRasterImage(Pic.Graphic).CreateIntfImage;
      try
        W := Img.Width;
        H := Img.Height;
        SetLength(ARGB, W * H);
        for j := 0 to H - 1 do
          for i := 0 to W - 1 do
          begin
            c := Img.Colors[i, j];
            ARGB[i + W * j] := (c.Red shr 8) shl 16 or (c.Green shr 8) shl 8 or (c.Blue shr 8);
          end;
        Result := True;
      finally
        Img.Free;
      end;
    except
      on E: Exception do AError := E.Message;
    end;
  finally
    Pic.Free;
  end;
end;

{ ------------------------------------------------------------- building --- }

constructor TI2MMainForm.Create(AOwner: TComponent);
const
  Icons: array[0..6] of string = ('open', 'tetmesh', 'run', 'stop', 'saveas', 'save', 'fit');
  MenuIconNames: array[0..2] of string = ('openimage', 'tetmesh', 'opencad');
var
  G: TBitmap;
  C, H: TPanel;
  k: Integer;
begin
  inherited Create(AOwner);   { the designed form, i2mmain.lfm }
  FRecent := TStringList.Create;
  FGrayMap := 0;   { jet }
  I2MPictureReaderHook := @ReadPictureLCL;   { (pictures through the LCL's TPicture) }
  FShapes := TI2MShapeDoc.Create;
  FShapeSel := -1;
  FSelPath := nil;
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
  for k := 0 to 2 do   { the Open menu's: an image, a mesh, shapes / CAD / PLC }
  begin
    G := I2MIcon(MenuIconNames[k], OpenMenuIcons.Width);
    if G <> nil then
    begin
      OpenMenuIcons.Add(G, nil);
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
  B2MChevron.Caption := MeshingChevron.Caption;
  SiamChevron.Caption := MeshingChevron.Caption;
  MeshingClose.Caption := #$C3#$97;         { U+00D7, a multiplication sign }
  DisplayClose.Caption := MeshingClose.Caption;
  ShapesClose.Caption := MeshingClose.Caption;
  B2MClose.Caption := MeshingClose.Caption;
  SiamClose.Caption := MeshingClose.Caption;
  for k := 0 to ComponentCount - 1 do
    if (Components[k] is TPanel) and (SectionBody(TPanel(Components[k])) <> nil) then
      PaintHead(TPanel(Components[k]), False);
  BuildMeshingSections;
  OpenSection('Mode');
  OpenSection('Crop box');
  for H in [SectB2MPathHead, SectSiamPathHead] do   { (their panels left hidden) }
    if not SectionBody(H).Visible then OpenHead(H);
  { the designed layout (Reset layout) and the saved one: once shown (the form's
    DPI scaling is applied after this constructor; bounds set before it would be
    scaled again -- a panel grew by the scale at every start) }
  Application.QueueAsyncCall(@ApplyLayout, 0);
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
  { gpu_brain2mesh and siamize: the current folder, then the PATH (FindTool); a path
    picked in a panel is kept in v2m.ini }
  B2MExeEdit.Text := FindTool('brain2mesh', 'V2M_BRAIN2MESH');
  SiamExeEdit.Text := FindTool('siamize', 'V2M_SIAMIZE');
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
  FreeAndNil(FRecent);
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
begin
  BuildOptionRows(Options, FEdits, FArgEdits, [ExeLabel, FormatLabel], @OptionChanged);
  BuildOptionRows(B2MOptions, FB2MEdits, FB2MArgEdits, [B2MExeLabel, B2MOutLabel], @ToolChanged);
  BuildOptionRows(SiamOptions, FSiamEdits, FSiamArgEdits, [SiamExeLabel, SiamOutLabel], @ToolChanged);
end;

{ a table's option rows, into the designed sections its groups name; the rows'
  left-hand captions (and ALefts) share one width, fitted }
procedure TI2MMainForm.BuildOptionRows(const AOpts: array of TI2MOption; var AEdits: TI2MControls;
  var AArgEdits: TI2MEdits; const ALefts: array of TControl; AOnChange: TNotifyEvent);
var
  Box: TPanel;   { the current section's body }
  Card: TPanel;  { the panel the rows go in }
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
    Box := SectionBody(SectionHead(ACaption, Card));
    if Box = nil then raise Exception.Create('i2mmain.lfm has no section "' + ACaption + '"');
  end;

begin
  Box := nil;
  Card := nil;
  if Length(ALefts) > 0 then Card := CardOf(ALefts[0]);   { (its panel: its sections only) }
  { the designed rows' captions share the fitted column }
  SetLength(Lefts, Length(ALefts));
  for i := 0 to High(ALefts) do Lefts[i] := ALefts[i];
  SetLength(AEdits, Length(AOpts));
  SetLength(AArgEdits, Length(AOpts));
  for i := 0 to High(AOpts) do
  begin
    if AOpts[i].Group <> '' then Heading(AOpts[i].Group);
    Row := NewRow;
    Row.Hint := AOpts[i].Flag + ': ' + AOpts[i].Hint;
    Row.ShowHint := True;
    case AOpts[i].Kind of
      okBool:
        begin
          C := TCheckBox.Create(Self);
          C.Parent := Row;
          C.Align := alClient;
          C.Caption := AOpts[i].Caption;
          C.BorderSpacing.Left := 6;
          C.OnChange := AOnChange;
          C.Hint := Row.Hint;
          C.ShowHint := True;
          AEdits[i] := C;
        end;
      okChoice:
        begin
          L := RowLabel(Row, AOpts[i].Caption);
          L.Hint := Row.Hint;
          Cb := TComboBox.Create(Self);
          Cb.Parent := Row;
          Cb.Align := alClient;
          Cb.Style := csDropDownList;
          Items := AOpts[i].Default.Split('|');
          for s in Items do Cb.Items.Add(s);
          Cb.ItemIndex := 0;
          Cb.BorderSpacing.Right := 4;
          Cb.OnChange := AOnChange;
          Cb.Hint := Row.Hint;
          Cb.ShowHint := True;
          AEdits[i] := Cb;
        end;
      okFlagArg:
        begin
          C := TCheckBox.Create(Self);
          C.Parent := Row;
          C.Align := alLeft;
          C.Width := 150;
          SetLength(Lefts, Length(Lefts) + 1);
          Lefts[High(Lefts)] := C;
          C.Caption := AOpts[i].Caption;
          C.BorderSpacing.Left := 6;
          C.OnChange := AOnChange;
          C.Hint := Row.Hint;
          C.ShowHint := True;
          AEdits[i] := C;
          E := TEdit.Create(Self);
          E.Parent := Row;
          E.Align := alClient;
          E.TextHint := AOpts[i].Default;
          E.BorderSpacing.Right := 4;
          E.OnChange := AOnChange;
          E.Hint := Row.Hint;
          E.ShowHint := True;
          AArgEdits[i] := E;
        end;
    else
      begin
        L := RowLabel(Row, AOpts[i].Caption);
        L.Hint := Row.Hint;
        E := TEdit.Create(Self);
        E.Parent := Row;
        E.Align := alClient;
        E.TextHint := AOpts[i].Default;
        E.BorderSpacing.Right := 4;
        E.OnChange := AOnChange;
        E.Hint := Row.Hint;
        E.ShowHint := True;
        AEdits[i] := E;
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
      if FVolFile <> '' then AList.Add(FVolFile) else AList.Add('<volume>');
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
        if (TComboBox(FEdits[i]).ItemIndex > 0) or ((Options[i].Flag = '--mode') and (Mode <> 'mesh')) then
        begin   { (the mode: as run -- mesh with a surface open is remesh) }
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

{ the Make choice as it stands }
function TI2MMainForm.ChosenMode: string;
var
  i: Integer;
begin
  Result := 'mesh';
  for i := 0 to High(Options) do
    if (Options[i].Flag = '--mode') and (FEdits[i] <> nil) then
      Result := ModeNames[Max(0, TComboBox(FEdits[i]).ItemIndex)];
end;

{ the mode run: the choice, except that mesh / surface -- which make a volume's or a
  shape design's -- with only a mesh or a surface open make the surface's: remesh
  (the whole mesher on it) / repair }
function TI2MMainForm.Mode: string;
begin
  Result := ChosenMode;
  if ((Result = 'mesh') or (Result = 'surface')) and (FVol.Nx = 0) and not FShapesOn and (FMeshFile <> '') then
    if Result = 'mesh' then Result := 'remesh' else Result := 'repair';
end;

function TI2MMainForm.MeshInput: Boolean;
begin   { (a shape design is read as its exact surface by cdt / remesh / repair: still the design) }
  Result := (Mode <> 'mesh') and (Mode <> 'surface') and not (FShapesOn and (Mode <> 'optimize'));
end;

function TI2MMainForm.InputMeshFile: string;
begin
  if FMeshSrc <> '' then Exit(FMeshSrc);   { (a CAD model / PLC: v2mesh reads it again, with the settings) }
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
  UpdateToolCommands;
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
  if B2MRunBtn <> nil then
  begin   { (brain2mesh, siamize: on the volume open) }
    B2MRunBtn.Enabled := (FVolFile <> '') and not FShapesOn and not Running;
    SiamRunBtn.Enabled := B2MRunBtn.Enabled;
    B2MStopBtn.Enabled := Running and (FJob = 1);
    SiamStopBtn.Enabled := Running and (FJob = 2);
  end;
  SyncMainMenu;
end;

{ -------------------------------------------------------------- display --- }

{ letters naming a direction for each axis (one of R/L, A/P, S/I each, as RAS, PSL) }
function ValidOrient(const S: string): Boolean;
var
  k, a: Integer;
  Used: array[0..2] of Boolean;
begin
  Result := Length(S) = 3;
  for k := 0 to 2 do Used[k] := False;
  for k := 1 to Length(S) do
  begin
    a := Pos(UpCase(S[k]), 'RAS') - 1;
    if a < 0 then a := Pos(UpCase(S[k]), 'LPI') - 1;
    if (a < 0) or Used[a] then Exit(False);
    Used[a] := True;
  end;
end;

{ the view's orientation: the file's, or the one chosen / typed in Display >
  Volume (display only: the image and the mesh turn together) }
procedure TI2MMainForm.ApplyOrientation;
var
  t: string;
begin
  t := UpperCase(Trim(OrientCombo.Text));
  if (OrientCombo.ItemIndex = 0) or not ValidOrient(t) then t := FFileOrient;
  FView.Orientation := t;
end;

procedure TI2MMainForm.SetOrientationText(const AText: string);
begin
  if SameText(AText, 'file') then OrientCombo.ItemIndex := 0
  else OrientCombo.Text := UpperCase(AText);
  OrientChanged(OrientCombo);
end;

procedure TI2MMainForm.OrientChanged(Sender: TObject);
var
  t: string;
begin
  if FUpdating then Exit;
  t := UpperCase(Trim(OrientCombo.Text));
  if (OrientCombo.ItemIndex <> 0) and not ValidOrient(t) then Exit;   { (still typing) }
  ApplyOrientation;
  if FVol.Nx > 0 then ResetView;   { (upright, from the front) }
end;

procedure TI2MMainForm.DisplayChanged(Sender: TObject);
begin
  if (FView = nil) or FUpdating then Exit;
  FView.ShowVolume := ShowVolCheck.Checked;
  FView.ShowMesh := ShowMeshCheck.Checked;
  FView.ShowEdges := ShowEdgesCheck.Checked;
  FView.Colormap := MapCombo.ItemIndex;
  { (picked for an intensity volume: the next one opens with it too) }
  if (Sender = MapCombo) and (FVol.Nx > 0) and not FVol.IsInteger and (FVol.Nc = 1) then
    FGrayMap := MapCombo.ItemIndex;
  FView.Style := StyleCombo.ItemIndex;
  FView.Opacity := OpacityTrack.Position / 100;
  FView.Threshold := FloorTrack.Position / 100;
  FView.MeshAlpha := MeshAlphaTrack.Position / 100;   { a uniform: no rebuild }
  UpdateFloorLabel;
  FView.Redraw;
end;

{ Hide below: the slider's value in the image's units, and the range it spans (the
  channel shown's: the volume's colour range) }
procedure TI2MMainForm.UpdateFloorLabel;
var
  Lo, Hi: Double;
begin
  if FVol.Nx = 0 then
  begin
    FloorLabel.Caption := 'Hide below (fraction of range)';
    Exit;
  end;
  Lo := FDispLo;
  Hi := FDispHi;
  if Hi <= Lo then Hi := Lo + 1;   { (as the view: a flat image) }
  FloorLabel.Caption := Format('Hide below: %.4g  (%d%% of %.4g .. %.4g)',
    [Lo + FloorTrack.Position / 100 * (Hi - Lo), FloorTrack.Position, Lo, Hi]);
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
        if (t <= High(FLabelNames)) and (FLabelNames[t] <> '') then n := n + ' (' + FLabelNames[t] + ')'
        else if (FMesh <> nil) and (FMesh.LabelName(t) <> '') then n := n + ' (' + FMesh.LabelName(t) + ')';   { (a shell) }
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

{ Display > Labels > Surfaces: inner label only -- the faces a ticked label shows }
procedure TI2MMainForm.InnerOnlyChanged(Sender: TObject);
begin
  if FMesh = nil then Exit;
  FMesh.InnerOnly := InnerOnlyCheck.Checked;
  FView.MeshChanged;
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

{ Display > Labels > Merge selected: the selected labels become the lowest of
  them, in what v2mesh meshes -- a label volume's voxels (into a NIfTI copy in
  the temporary folder, the input from then on), a probability map's channel
  labels (--tpm-map); the names joined with '+' }
procedure TI2MMainForm.MergeLabelsClick(Sender: TObject);
var
  Sel, Map: TI2MIntegers;
  k, t, lo, i: Integer;
  nv: Int64;
  IsSel: array of Boolean;
  Joined, Err, Txt, Out_: string;

  function IntListText(const A: TI2MIntegers): string;
  var
    j: Integer;
  begin
    Result := '';
    for j := 0 to High(A) do Result := Result + IfThen(j > 0, ', ', '') + IntToStr(A[j]);
  end;

begin
  Sel := nil;
  for k := 0 to LabelList.Items.Count - 1 do
    if LabelList.Selected[k] then
    begin
      SetLength(Sel, Length(Sel) + 1);
      Sel[High(Sel)] := PtrInt(LabelList.Items.Objects[k]);
    end;
  if Length(Sel) < 2 then
  begin
    Log('merge: select two labels or more (Ctrl / Shift + click)');
    Exit;
  end;
  if Running then
  begin
    Log('v2mesh is running; not merging');
    Exit;
  end;
  if FVol.Nx = 0 then
  begin
    Log('merge: open a label volume or a probability map (a mesh''s labels are not merged)');
    Exit;
  end;
  lo := Sel[0];
  for t in Sel do lo := Min(lo, t);
  SetLength(IsSel, 1);
  for t in Sel do
  begin
    if t > High(IsSel) then SetLength(IsSel, t + 1);
    IsSel[t] := True;
  end;
  Joined := '';
  for t in Sel do
    if (t <= High(FLabelNames)) and (FLabelNames[t] <> '') then
      Joined := Joined + IfThen(Joined <> '', '+', '') + FLabelNames[t];
  if FVol.Nc > 1 then
  begin   { a probability map: its channels' labels }
    if ChannelCombo.ItemIndex > 0 then
    begin
      Log('merge: show the argmax labels (Channel) to merge a probability map''s labels');
      Exit;
    end;
    if not I2MChannelMap(FVol, OptionText('--tpm-map'), OptionText('--tpm-exterior'), Map, Err) then
    begin
      Log('merge: ' + Err);
      Exit;
    end;
    Txt := '';
    for i := 0 to High(Map) do
    begin
      if (Map[i] <= High(IsSel)) and IsSel[Map[i]] then Map[i] := lo;
      Txt := Txt + IfThen(i > 0, ',', '') + IntToStr(Map[i]);
    end;
    SetOption('--tpm-map', Txt);
    Log(Format('merged labels %s into %d%s: --tpm-map %s', [IntListText(Sel), lo,
      IfThen(Joined <> '', ' (' + Joined + ')', ''), Txt]));
  end
  else
  begin   { a label volume: its voxels }
    if not FVol.IsInteger then
    begin
      Log('merge: the volume is not a label volume');
      Exit;
    end;
    nv := Int64(FVol.Nx) * FVol.Ny * FVol.Nz;
    k := 0;
    for i := 0 to nv - 1 do
    begin
      t := Round(FVol.Data[i]);
      if (t >= 0) and (t <= High(IsSel)) and IsSel[t] and (t <> lo) then
      begin
        FVol.Data[i] := lo;
        Inc(k);
      end;
    end;
    if lo >= Length(FVol.LabelNames) then SetLength(FVol.LabelNames, lo + 1);
    if Joined <> '' then FVol.LabelNames[lo] := Joined;
    for t in Sel do
      if (t <> lo) and (t <= High(FVol.LabelNames)) then FVol.LabelNames[t] := '';
    Out_ := IncludeTrailingPathDelimiter(GetTempDir(False)) + 'v2m_' +
      ChangeFileExt(ChangeFileExt(ExtractFileName(IfThen(FPicFile <> '', FPicFile, FVolFile)), ''), '') + '_merged.nii';
    if not I2MSaveNifti(Out_, FVol, Err) then
    begin
      Log('merge: could not write ' + Out_ + ': ' + Err);
      Exit;
    end;
    FVolFile := Out_;
    Log(Format('merged labels %s into %d%s: %d voxels; meshed from %s', [IntListText(Sel), lo,
      IfThen(Joined <> '', ' (' + Joined + ')', ''), k, Out_]));
  end;
  ShowChannel;
  UpdateCommand;
end;

{ --merge L1,L2,..: those labels selected, then merged (as the button) }
procedure TI2MMainForm.MergeLabels(const AList: string);
var
  k, t: Integer;
  Want: TStringArray;
begin
  Want := AList.Split([',', ' '], TStringSplitOptions.ExcludeEmpty);
  LabelList.ClearSelection;
  for k := 0 to LabelList.Items.Count - 1 do
    for t := 0 to High(Want) do
      if StrToIntDef(Trim(Want[t]), -1) = PtrInt(LabelList.Items.Objects[k]) then LabelList.Selected[k] := True;
  MergeLabelsClick(nil);
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
  UpdateFloorLabel;
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
    { a label map's names: its LabelTable, keyed by label value (siamize's) }
    if FDispIsLabel then FLabelNames := Copy(FVol.LabelNames) else FLabelNames := nil;
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
    if FShapes.Count > 0 then SetSelPath([0]) else SetSelPath(nil);
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
  if Result and not FPair then ClearMesh;   { (another input: the old mesh is not of it) }
  if Result then
  begin
    RememberDir(AFileName);
    ShowOnly('volume');   { (an image opened: it, not the mesh; one click brings the mesh back) }
  end;
  if not Result then
  begin
    Log('could not read ' + AFileName + ': ' + Err);
    StatusBar.SimpleText := 'could not read ' + ExtractFileName(AFileName);
    Exit;
  end;
  FVolFile := ExpandFileName(AFileName);
  FPicFile := '';
  if I2MIsPicture(AFileName) then
  begin   { a picture: v2mesh meshes it (2-D) from a one-slice NIfTI copy }
    FPicFile := ExpandFileName(AFileName);
    FVolFile := IncludeTrailingPathDelimiter(GetTempDir(False)) + 'v2m_' +
      ChangeFileExt(ExtractFileName(AFileName), '') + '.nii';
    if not I2MSaveNifti(FVolFile, FVol, Err) then
    begin
      Log('could not write ' + FVolFile + ': ' + Err);
      FVolFile := ExpandFileName(AFileName);
    end
    else
      Log(Format('%s: a picture as %s, %s; meshed (2-D) from %s', [ExtractFileName(AFileName),
        PictCombo.Items[Ord(I2MPictureMode)] + IfThen(I2MPictureMode in [pmBinary, pmBinaryDark],
          Format(' at %d%s', [I2MPictureLastThreshold, IfThen(I2MPictureThreshold < 0, ' (Otsu)', '')]), ''),
        IfThen(FVol.IsInteger, Format('%d labels', [Round(FVol.High) + 1]), 'an intensity volume (set --thresholds)'),
        FVolFile]));
  end;
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
    MapCombo.ItemIndex := FGrayMap;
    FloorTrack.Position := 5;
  end;
  FUpdating := False;
  DisplayChanged(nil);
  if FVol.Oriented then
    FFileOrient := I2MAxisLetter(FVol.Affine, 0) + I2MAxisLetter(FVol.Affine, 1) + I2MAxisLetter(FVol.Affine, 2)
  else
    FFileOrient := '';
  FUpdating := True;   { (a new image: its own orientation again) }
  OrientCombo.Items[0] := 'from the file (' + IfThen(FFileOrient = '', 'none', FFileOrient) + ')';
  OrientCombo.ItemIndex := 0;
  FUpdating := False;
  ApplyOrientation;
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

procedure TI2MMainForm.ClearMesh;
begin
  if FMesh = nil then Exit;
  FView.SetMesh(nil);
  FreeAndNil(FMesh);
  FMeshFile := '';
  FMeshSrc := '';
  FillLabels;
  UpdateStats;
  UpdateCommand;
  UpdateButtons;
end;

procedure TI2MMainForm.ClearVolumeData;
begin
  if FVol.Nx = 0 then Exit;
  FView.ClearVolume;
  FVol := Default(TI2MVolume);
  FVolFile := '';
  FFileOrient := '';
  FView.Orientation := '';
  UpdateFloorLabel;
  FillLabels;
  UpdateCommand;
  UpdateButtons;
end;

function TI2MMainForm.LoadMesh(const AFileName: string; AReset: Boolean): Boolean;
var
  M: TI2MMesh;
  T0: QWord;
  First: Boolean;
begin
  T0 := GetTickCount64;
  M := TI2MMesh.Create;
  M.InnerOnly := InnerOnlyCheck.Checked;
  Result := M.LoadFromFile(AFileName);
  if not Result then
  begin
    Log('could not read ' + AFileName + ': ' + M.Error);
    M.Free;
    Exit;
  end;
  if AReset and not FPair then
  begin   { (a mesh opened: the old volume, or a shape design's drawing, is not of it) }
    ClearVolumeData;
    if FShapesOn then
    begin
      FShapesOn := False;
      FView.ClearShapes;
    end;
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
  FMeshSrc := '';
  if AReset then RememberDir(AFileName);   { (not a run's result, in the temporary folder) }
  { a dense mesh's wireframe is a solid colour at any ordinary zoom }
  if First or AReset then   { (a mesh opened: it, not the image; a run's result keeps the toggles) }
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
    D.Title := 'Open a volume';
    D.Filter := 'Volumes and shapes (*.nii;*.nii.gz;*.jnii;*.bnii;*.json)|*.nii;*.nii.gz;*.gz;*.jnii;*.bnii;*.json|' +
      '2-D pictures (*.png;*.bmp;*.jpg;*.gif;*.tif;*.pgm;*.xpm;*.ico)|*.png;*.bmp;*.jpg;*.jpeg;*.gif;*.tif;*.tiff;*.pbm;*.pgm;*.ppm;*.pnm;*.xpm;*.ico|' +
      'All files|*';
    if D.Execute and LoadImage(D.FileName) then AddRecent(D.FileName);
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
    if D.Execute and LoadMesh(D.FileName) then AddRecent(D.FileName);
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
    3: OpenSection('Labels');
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

{ the scene framed and centred in the whole view (the floating panels, pinned
  or not, are over it: not left room for) }
procedure TI2MMainForm.FitView;
begin
  FView.FitView(0, 0);
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

{ what a dropped / named file is: 1 an image (or shapes), 2 a mesh, 3 a CAD
  model or a PLC, 0 none of them }
function FileKind(const AFileName: string): Integer;
var
  n: string;
begin
  n := LowerCase(ExtractFileName(AFileName));
  if n.EndsWith('.jmsh') or n.EndsWith('.bmsh') or n.EndsWith('.off') or n.EndsWith('.stl') then Exit(2);
  if n.EndsWith('.step') or n.EndsWith('.stp') or n.EndsWith('.poly') or n.EndsWith('.smesh') then Exit(3);
  if n.EndsWith('.nii') or n.EndsWith('.nii.gz') or n.EndsWith('.jnii') or
     n.EndsWith('.bnii') or n.EndsWith('.json') or I2MIsPicture(n) then Exit(1);
  Result := 0;
end;

procedure TI2MMainForm.FormDropFiles(Sender: TObject; const FileNames: array of string);
var
  f: string;
  k: Integer;
begin
  { volumes first, so a mesh dropped with its volume lands on it (both kept) }
  FPair := False;
  for f in FileNames do
    if FileKind(f) = 1 then
      for k := 0 to High(FileNames) do
        FPair := FPair or (FileKind(FileNames[k]) in [2, 3]);
  for k := 1 to 3 do
    for f in FileNames do
      if FileKind(f) = k then
      begin
        if Running and (k <> 2) then
        begin
          Log('v2mesh is running; not opening ' + ExtractFileName(f));
          Continue;
        end;
        if OpenAny(f) then AddRecent(f);
      end;
  FPair := False;
  for f in FileNames do
    if FileKind(f) = 0 then
      Log('not a volume (.nii .nii.gz .jnii .bnii .json, a picture), a mesh (.jmsh .bmsh .off .stl), ' +
        'a CAD model (.step .stp) or a PLC (.poly .smesh): ' + f);
  BringToFront;
end;

function TI2MMainForm.OpenAny(const AFileName: string): Boolean;
begin
  case FileKind(AFileName) of
    1: Result := LoadImage(AFileName);
    2: Result := LoadMesh(AFileName);
    3: Result := LoadCad(AFileName);
  else
    begin
      Log('not a file v2m opens: ' + AFileName);
      Result := False;
    end;
  end;
end;

const
  ImageFilter = '*.nii;*.nii.gz;*.gz;*.jnii;*.bnii';
  PictureFilter = '*.png;*.bmp;*.jpg;*.jpeg;*.gif;*.tif;*.tiff;*.pbm;*.pgm;*.ppm;*.pnm;*.xpm;*.ico';
  MeshFilter = '*.jmsh;*.bmsh;*.off;*.stl';
  CadFilter = '*.json;*.step;*.stp;*.poly;*.smesh';

{ the toolbar's Open: any kind, by its suffix (the arrow's menu: one kind) }
procedure TI2MMainForm.OpenAnyClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Open';
    D.Filter := 'All v2m files|' + ImageFilter + ';' + PictureFilter + ';' + MeshFilter + ';' + CadFilter + '|' +
      'Volumes (.nii .jnii .bnii)|' + ImageFilter + '|2-D pictures|' + PictureFilter + '|' +
      'Meshes and surfaces (.jmsh .bmsh .off .stl)|' + MeshFilter + '|' +
      'Shapes, CAD models, PLCs (.json .step .stp .poly .smesh)|' + CadFilter + '|All files|*';
    if D.Execute then
      if Running and (FileKind(D.FileName) <> 2) then
        Log('v2mesh is running; not opening ' + ExtractFileName(D.FileName))
      else if OpenAny(D.FileName) then
        AddRecent(D.FileName);
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.OpenCadClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  UseLastDir(D);
  try
    D.Title := 'Open shape constructs, a CAD model or a PLC';
    D.Filter := 'Shapes, CAD models, PLCs (.json .step .stp .poly .smesh)|' + CadFilter + '|' +
      'CAD models (.step .stp)|*.step;*.stp|TetGen PLCs (.poly .smesh)|*.poly;*.smesh|' +
      'Shape constructs (.json)|*.json|All files|*';
    if D.Execute and OpenAny(D.FileName) then AddRecent(D.FileName);
  finally
    D.Free;
  end;
end;

{ ---------------------------------------------------------- recent files --- }

const
  MaxRecent = 10;

{ a file opened from a dialog or dropped: first in File > Open recent }
procedure TI2MMainForm.AddRecent(const AFileName: string);
var
  f: string;
  k: Integer;
begin
  f := ExpandFileName(AFileName);
  k := FRecent.IndexOf(f);
  if k >= 0 then FRecent.Delete(k);
  FRecent.Insert(0, f);
  while FRecent.Count > MaxRecent do FRecent.Delete(FRecent.Count - 1);
  SaveRecent;
  BuildRecentMenu;
end;

{ (written at once: kept if v2m does not close cleanly) }
procedure TI2MMainForm.SaveRecent;
var
  Ini: TIniFile;
  k: Integer;
begin
  try
    ForceDirectories(ExtractFilePath(LayoutFile));
    Ini := TIniFile.Create(LayoutFile);
    try
      Ini.EraseSection('Recent');
      for k := 0 to FRecent.Count - 1 do Ini.WriteString('Recent', 'File' + IntToStr(k + 1), FRecent[k]);
    finally
      Ini.Free;
    end;
  except
    on E: Exception do Log('could not save the recent files: ' + E.Message);
  end;
end;

procedure TI2MMainForm.BuildRecentMenu;
var
  M: TMenuItem;
  k: Integer;
begin
  MainRecent.Clear;
  for k := 0 to FRecent.Count - 1 do
  begin
    M := TMenuItem.Create(MainRecent);
    { (&1 .. &9, then 1&0: the accelerators; the folder after the name) }
    M.Caption := IfThen(k < 9, '&' + IntToStr(k + 1), '1&0') + '  ' + ExtractFileName(FRecent[k]) + '   (' +
      ExtractFileDir(FRecent[k]) + ')';
    M.Tag := k;
    M.OnClick := @RecentClick;
    MainRecent.Add(M);
  end;
  if FRecent.Count > 0 then
  begin
    M := TMenuItem.Create(MainRecent);
    M.Caption := '-';
    MainRecent.Add(M);
  end;
  M := TMenuItem.Create(MainRecent);
  M.Caption := '&Clear the list';
  M.Enabled := FRecent.Count > 0;
  M.OnClick := @RecentClearClick;
  MainRecent.Add(M);
end;

procedure TI2MMainForm.RecentClick(Sender: TObject);
var
  f: string;
begin
  if not (Sender is TMenuItem) or (TMenuItem(Sender).Tag >= FRecent.Count) then Exit;
  f := FRecent[TMenuItem(Sender).Tag];
  if not FileExists(f) then
  begin
    Log('no longer there: ' + f);
    FRecent.Delete(TMenuItem(Sender).Tag);
    SaveRecent;
    BuildRecentMenu;
    Exit;
  end;
  if Running and (FileKind(f) <> 2) then
    Log('v2mesh is running; not opening ' + ExtractFileName(f))
  else if OpenAny(f) then
    AddRecent(f);
end;

procedure TI2MMainForm.RecentClearClick(Sender: TObject);
begin
  FRecent.Clear;
  SaveRecent;
  BuildRecentMenu;
end;

{ ------------------------------------------------------------- pictures --- }

{ the conversion of the pictures changed: the open picture converted again }
procedure TI2MMainForm.PictChanged(Sender: TObject);
var
  t: Integer;
  PM: TI2MPictureMode;
begin
  PM := TI2MPictureMode(Max(0, PictCombo.ItemIndex));
  if not TryStrToInt(Trim(PictThreshEdit.Text), t) or (t < 0) or (t > 255) then
  begin
    if Trim(PictThreshEdit.Text) <> '' then Log('binary threshold: want 0 .. 255 (blank: Otsu); Otsu''s used');
    t := -1;
  end;
  PictThreshEdit.Enabled := PM in [pmBinary, pmBinaryDark];
  PictThreshLabel.Enabled := PictThreshEdit.Enabled;
  if (PM = I2MPictureMode) and (t = I2MPictureThreshold) then Exit;
  I2MPictureMode := PM;
  I2MPictureThreshold := t;
  if (FPicFile <> '') and FileExists(FPicFile) and not Running then LoadImage(FPicFile);
end;

function TI2MMainForm.LoadCad(const AFileName: string): Boolean;
var
  Tmp, Sz, Output: string;
  Args: array of string;
  Status: Integer;
  L: TStringList;
  k: Integer;
begin
  Result := False;
  if Running then
  begin
    Log('v2mesh is running; not opening ' + ExtractFileName(AFileName));
    Exit;
  end;
  Tmp := IncludeTrailingPathDelimiter(GetTempDir(False)) + 'v2m_' +
    ChangeFileExt(ExtractFileName(AFileName), '') + '.jmsh';
  Args := ['--mode', 'convert', '-i', AFileName, '-o', Tmp];
  Sz := OptionText('--size');   { (the tessellation's edge cap, as the mesh modes will use it) }
  if Sz <> '' then Args := Concat(Args, ['--size', Sz]);
  Log(ExtractFileName(AFileName) + ': read by ' + ExtractFileName(ExeEdit.Text) + ' --mode convert ...');
  StatusBar.SimpleText := 'reading ' + ExtractFileName(AFileName) + ' ...';
  Application.ProcessMessages;
  Screen.Cursor := crHourGlass;
  try
    Status := -1;
    try
      RunCommandInDir(GetCurrentDir, ExeEdit.Text, Args, Output, Status, [poStderrToOutPut, poNoConsole]);
    except
      on E: Exception do Output := 'could not start ' + ExeEdit.Text + ': ' + E.Message;
    end;
  finally
    Screen.Cursor := crDefault;
  end;
  L := TStringList.Create;
  try
    L.Text := Output;
    for k := 0 to L.Count - 1 do
      if Trim(L[k]) <> '' then Log(L[k]);
  finally
    L.Free;
  end;
  if (Status <> 0) or not FileExists(Tmp) then
  begin
    Log('could not read ' + AFileName);
    StatusBar.SimpleText := 'could not read ' + ExtractFileName(AFileName);
    Exit;
  end;
  if not LoadMesh(Tmp, True) then Exit;
  FMeshSrc := ExpandFileName(AFileName);   { (after LoadMesh, which clears it) }
  RememberDir(AFileName);
  if (ChosenMode = 'mesh') or (ChosenMode = 'surface') then
    SetOption('--mode', 'cdt');   { (its faces kept; remesh / repair also read it) }
  Caption := 'v2m - ' + ExtractFileName(AFileName);
  StatusBar.SimpleText := ExtractFileName(AFileName) + ': ' + StatusBar.SimpleText;
  UpdateCommand;
  UpdateButtons;
  Result := True;
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
    else Log('open a volume first');
    Exit;
  end;
  FreeAndNil(FProc);
  if MeshInput and (FMeshSrc = '') and (InputMeshFile <> FMeshFile) then
    if not CopyMesh(FMeshFile, InputMeshFile) then
    begin
      L.Free;
      Log('could not copy ' + FMeshFile + ' to ' + InputMeshFile);
      Exit;
    end;
  RunJob(0, ExeEdit.Text, L, FOutFile, CmdEdit.Text);
  L.Free;
end;

{ a program started (v2mesh, brain2mesh, siamize: ATool 0, 1, 2), its output in
  the log; when it ends, Finished shows AOut }
procedure TI2MMainForm.RunJob(ATool: Integer; const AExe: string; AArgs: TStringList; const AOut, ACmd: string);
begin
  FreeAndNil(FProc);
  if FileExists(AOut) then DeleteFile(AOut);   { (a stale result is not shown as this one's) }
  FJob := ATool;
  FJobOut := AOut;
  FJobName := ChangeFileExt(ExtractFileName(AExe), '');
  FProc := TProcess.Create(nil);
  FProc.Executable := AExe;
  FProc.Parameters.Assign(AArgs);
  FProc.Options := [poUsePipes, poStderrToOutPut, poNoConsole];
  FPending := '';
  Log('');
  Log('$ ' + ACmd);
  try
    FProc.Execute;
  except
    on E: Exception do
    begin
      Log('could not start ' + AExe + ': ' + E.Message);
      FreeAndNil(FProc);
      UpdateButtons;
      Exit;
    end;
  end;
  FStarted := Now;
  FTimer.Enabled := True;
  StatusBar.SimpleText := FJobName + ' is running...';
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
    Log(FJobName + ' stopped');
  end;
end;

{ -------------------------------------------------- brain2mesh / siamize --- }

{ a program: $AEnv (a path) if set, else in the current folder, else in v2m's own
  folder (a package shipping them together), else on the PATH; else its bare name
  (set it in the panel) }
function TI2MMainForm.FindTool(const ABin, AEnv: string): string;
var
  BinName, c: string;
begin
  {$IFDEF WINDOWS}BinName := ABin + '.exe';{$ELSE}BinName := ABin;{$ENDIF}
  c := GetEnvironmentVariable(AEnv);
  if (c <> '') and FileExists(c) then Exit(ExpandFileName(c));
  for c in [IncludeTrailingPathDelimiter(GetCurrentDir), ExtractFilePath(ExpandFileName(ParamStr(0)))] do
    if FileExists(c + BinName) and not DirectoryExists(c + BinName) then Exit(ExpandFileName(c + BinName));
  Result := ExeSearch(BinName, GetEnvironmentVariable('PATH'));
  if Result = '' then Result := BinName;
end;

{ the file a tool writes: its Output, else one in the temporary folder (brain2mesh:
  a binary JMesh; siamize: a binary JNIfTI, whose label table names the classes) }
function TI2MMainForm.ToolOut(ATool: Integer): string;
begin
  if ATool = 1 then Result := Trim(B2MOutEdit.Text) else Result := Trim(SiamOutEdit.Text);
  if Result <> '' then Exit;
  Result := IncludeTrailingPathDelimiter(GetTempDir(False)) +
    Format('v2m-%d-%s', [GetProcessID, IfThen(ATool = 1, 'b2m.bmsh', 'seg.bnii')]);
end;

{ the tool's arguments: -i the volume open, -o its Output, then the panel's
  options; False when there is no volume file to give it }
function TI2MMainForm.ToolArgs(ATool: Integer; out AList: TStringList): Boolean;
var
  i: Integer;
  v: string;
  Extra: TStringList;
  Opts: array of TI2MOption;
  Ed: TI2MControls;
  Ar: TI2MEdits;
begin
  AList := TStringList.Create;
  Result := (FVolFile <> '') and not FShapesOn and FileExists(FVolFile);
  AList.Add('-i');
  if Result then AList.Add(FVolFile) else AList.Add(IfThen(ATool = 1, '<label map>', '<T1 volume>'));
  AList.Add('-o');
  AList.Add(ToolOut(ATool));
  if ATool = 1 then
  begin
    SetLength(Opts, Length(B2MOptions));
    for i := 0 to High(B2MOptions) do Opts[i] := B2MOptions[i];
    Ed := FB2MEdits;
    Ar := FB2MArgEdits;
  end
  else
  begin
    SetLength(Opts, Length(SiamOptions));
    for i := 0 to High(SiamOptions) do Opts[i] := SiamOptions[i];
    Ed := FSiamEdits;
    Ar := FSiamArgEdits;
  end;
  for i := 0 to High(Opts) do
    if i <= High(Ed) then
      case Opts[i].Kind of
        okBool:
          if TCheckBox(Ed[i]).Checked then AList.Add(Opts[i].Flag);
        okChoice:
          if TComboBox(Ed[i]).ItemIndex > 0 then
          begin
            AList.Add(Opts[i].Flag);
            AList.Add(TComboBox(Ed[i]).Text);
          end;
        okFlagArg:
          if TCheckBox(Ed[i]).Checked then
          begin
            AList.Add(Opts[i].Flag);
            v := Trim(Ar[i].Text);
            if v <> '' then AList.Add(v);
          end;
      else
        begin
          v := Trim(TEdit(Ed[i]).Text);
          if v <> '' then
          begin
            AList.Add(Opts[i].Flag);
            AList.Add(v);
          end;
        end;
      end;
  if ATool = 1 then v := Trim(B2MExtraEdit.Text) else v := Trim(SiamExtraEdit.Text);
  if v <> '' then
  begin
    Extra := TStringList.Create;
    try
      CommandToList(v, Extra);
      AList.AddStrings(Extra);
    finally
      Extra.Free;
    end;
  end;
end;

{ the panels' command lines, and what their input is }
procedure TI2MMainForm.UpdateToolCommands;
var
  L: TStringList;
  t: Integer;
  s, a, inp: string;
begin
  if (B2MCmdEdit = nil) or (SiamCmdEdit = nil) then Exit;
  for t := 1 to 2 do
  begin
    ToolArgs(t, L);
    try
      if t = 1 then s := B2MExeEdit.Text else s := SiamExeEdit.Text;
      for a in L do
        if (Pos(' ', a) > 0) or (a = '') then s := s + ' "' + a + '"' else s := s + ' ' + a;
      if t = 1 then B2MCmdEdit.Text := s else SiamCmdEdit.Text := s;
    finally
      L.Free;
    end;
  end;
  if (FVolFile <> '') and not FShapesOn then inp := ExtractFileName(FVolFile) else inp := '';
  if inp <> '' then
  begin
    B2MInLabel.Caption := 'Input: ' + inp + ' (the volume open: a label map or a TPM)';
    SiamInLabel.Caption := 'Input: ' + inp + ' (the volume open: a T1 head MRI)';
  end
  else
  begin
    B2MInLabel.Caption := 'Input: open a label map (or run siamize) first';
    SiamInLabel.Caption := 'Input: open a T1 head MRI (NIfTI / JNIfTI) first';
  end;
end;

procedure TI2MMainForm.ToolChanged(Sender: TObject);
begin
  UpdateToolCommands;
  UpdateButtons;
end;

procedure TI2MMainForm.ToolBrowseClick(Sender: TObject);
var
  O: TOpenDialog;
  S: TSaveDialog;
  E: TEdit;
begin
  if (Sender = B2MExeBrowse) or (Sender = SiamExeBrowse) then
  begin
    if Sender = B2MExeBrowse then E := B2MExeEdit else E := SiamExeEdit;
    O := TOpenDialog.Create(Self);
    try
      O.FileName := E.Text;
      if O.Execute then E.Text := O.FileName;
    finally
      O.Free;
    end;
    Exit;
  end;
  if Sender = B2MOutBrowse then E := B2MOutEdit else E := SiamOutEdit;
  S := TSaveDialog.Create(Self);
  try
    UseLastDir(S);
    S.Options := S.Options + [ofOverwritePrompt];
    if Sender = B2MOutBrowse then
    begin
      S.Filter := 'Binary JMesh (*.bmsh)|*.bmsh|JMesh (*.jmsh)|*.jmsh';
      S.DefaultExt := '.bmsh';
    end
    else
    begin
      S.Filter := 'Binary JNIfTI (*.bnii)|*.bnii|NIfTI (*.nii.gz)|*.nii.gz|JNIfTI (*.jnii)|*.jnii';
      S.DefaultExt := '.bnii';
    end;
    S.FileName := E.Text;
    if S.Execute then E.Text := S.FileName;
  finally
    S.Free;
  end;
end;

{ Run brain2mesh / siamize (the panels' buttons; the Mesh menu: Tag 1 / 2) }
procedure TI2MMainForm.ToolRunClick(Sender: TObject);
begin
  if (Sender = SiamRunBtn) or ((Sender is TMenuItem) and (TMenuItem(Sender).Tag = 2)) then RunTool(2) else RunTool(1);
end;

procedure TI2MMainForm.SetToolArgs(ATool: Integer; const AArgs: string);
begin
  if ATool = 1 then B2MExtraEdit.Text := AArgs else SiamExtraEdit.Text := AArgs;
end;

procedure TI2MMainForm.ShowToolPanel(ATool: Integer);
var
  C: TPanel;
begin
  if ATool = 1 then C := B2MCard else C := SiamCard;
  C.Visible := True;
  C.BringToFront;
  KeepInView(C);
  UpdatePanelsMenu;
end;

procedure TI2MMainForm.RunTool(ATool: Integer);
var
  t: Integer;
  L: TStringList;
begin
  if Running then Exit;
  t := ATool;
  if not ToolArgs(t, L) then
  begin
    L.Free;
    if t = 1 then Log('brain2mesh: open a label map (or run siamize on a T1 MRI) first')
    else Log('siamize: open a T1-weighted head MRI (NIfTI / JNIfTI) first');
    Exit;
  end;
  UpdateToolCommands;
  try
    if t = 1 then RunJob(1, B2MExeEdit.Text, L, ToolOut(1), B2MCmdEdit.Text)
    else RunJob(2, SiamExeEdit.Text, L, ToolOut(2), SiamCmdEdit.Text);
  finally
    L.Free;
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
    StatusBar.SimpleText := Format('%s is running... %.1f s', [FJobName, (Now - FStarted) * 86400]);
end;

procedure TI2MMainForm.Finished;
var
  Code: Integer;
  Secs: Double;
begin
  if FPending <> '' then
  begin
    Log(FPending);
    FPending := '';
  end;
  Code := FProc.ExitStatus;
  Secs := (Now - FStarted) * 86400;
  Log(Format('%s finished in %.1f s, exit code %d', [FJobName, Secs, Code]));
  if (Code = 0) and FileExists(FJobOut) then
  begin
    if FJob = 2 then
    begin   { siamize: its segmentation, the volume now -- what brain2mesh / v2mesh mesh }
      if LoadImage(FJobOut) then
        Log('the segmentation is shown, and is the input of brain2mesh and v2mesh (' + FJobOut + ')');
    end
    else
    begin   { v2mesh, brain2mesh: the mesh, in the image's voxels }
      LoadMesh(FJobOut, False);
      ShowOnly('mesh');
    end;
  end
  else if Code = 0 then
    StatusBar.SimpleText := Format('%s wrote no %s; see the log', [FJobName, ExtractFileName(FJobOut)])
  else
    StatusBar.SimpleText := Format('%s failed (exit code %d) after %.1f s; see the log', [FJobName, Code, Secs]);
  if (Code = 0) and FileExists(FJobOut) then   { (what was loaded, and how long the run took) }
    StatusBar.SimpleText := StatusBar.SimpleText + Format('  --  %s %.1f s', [FJobName, Secs]);
  UpdateButtons;
end;

{ first shown: centred on the monitor under the pointer (the one it was started
  from), smaller if it would not fit its work area -- LCL's poScreenCenter centres
  it on the whole desktop, across monitors }
procedure TI2MMainForm.FormShow(Sender: TObject);
var
  M: TMonitor;
  R: TRect;
  W, H: Integer;
begin
  if FCentred then Exit;
  FCentred := True;
  M := Screen.MonitorFromPoint(Mouse.CursorPos);
  if M = nil then M := Screen.PrimaryMonitor;
  if M = nil then Exit;
  R := M.WorkareaRect;
  if (R.Right <= R.Left) or (R.Bottom <= R.Top) then R := M.BoundsRect;
  W := Min(Width, (R.Right - R.Left) * 95 div 100);
  H := Min(Height, (R.Bottom - R.Top) * 95 div 100);
  SetBounds(R.Left + (R.Right - R.Left - W) div 2, R.Top + (R.Bottom - R.Top - H) div 2, W, H);
end;

procedure TI2MMainForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if Running then FProc.Terminate(1);
  SaveLayout;
  CloseAction := caFree;
  { (v2m.lpr creates the form itself, not by Application.CreateForm: it is no
    MainForm, and its close would not end the application -- the event loop
    would go on with no window) }
  Application.Terminate;
end;

{ ---------------------------------------------------------------- cards --- }

const
  Snap = 8;
  { v2m.ini's layout: a file of another version (another set of panels, or
    pixels not at 96 dpi) is ignored, and replaced when the window closes }
  LayoutVersion = 4;   { (4: the corners, the pins) }   { a card dragged this close to the view's edge sticks to it }

function TI2MMainForm.AllCards: TI2MPanels;
begin
  Result := [MeshingCard, DisplayCard, ShapesCard, B2MCard, SiamCard];
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
  h, bottom, k: Integer;
begin
  B := CardBody(ACard);
  if (B = nil) or (B.Visible = not ACollapsed) then Exit;
  Chev := TLabel(FindComponent(ACard.Name.Replace('Card', 'Chevron')));
  bottom := ACard.Top + ACard.Height;
  for k := 0 to ACard.ControlCount - 1 do   { (a footer -- the tool panels' Run / Stop -- folds with the body) }
    if ACard.Controls[k].Align = alBottom then ACard.Controls[k].Visible := not ACollapsed;
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
  k := CardIndex(ACard);
  if ACollapsed and (k >= 0) and (k <= High(FHome)) then   { back where it was put }
    if akBottom in ACard.Anchors then ACard.SetBounds(FHome[k].X, FHome[k].Y - ACard.Height, ACard.Width, ACard.Height)
    else ACard.SetBounds(FHome[k].X, FHome[k].Y, ACard.Width, ACard.Height);
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
  else
  begin
    SetHome(C);   { (its new place) }
    if FView <> nil then FView.Redraw;   { what the card uncovered }
  end;
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
{ (ACard: only its sections -- the tool panels share captions, Program and Files..) }
function TI2MMainForm.SectionHead(const ACaption: string; ACard: TPanel): TPanel;
var
  k: Integer;
begin
  for k := 0 to ComponentCount - 1 do
    if (Components[k] is TPanel) and (SectionBody(TPanel(Components[k])) <> nil) and
       SameText(HeadText(TPanel(Components[k])), ACaption) and
       ((ACard = nil) or (CardOf(TPanel(Components[k])) = ACard)) then
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
  MenuB2M.Checked := B2MCard.Visible;
  MenuSiam.Checked := SiamCard.Visible;
end;

procedure TI2MMainForm.ViewMenuClick(Sender: TObject);
var
  C: TPanel;
  k, i: Integer;
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
    5: C := B2MCard;
    6: C := SiamCard;
  else
    begin   { Reset layout: the designed places, all shown and open }
      if Length(FDefaults) < Length(AllCards) then Exit;   { (not yet known: before the form is shown) }
      for k := 0 to High(AllCards) do
      begin
        C := AllCards[k];
        C.BoundsRect := DefaultRect(k);
        SetHome(C);
        C.Tag := 0;
        C.Visible := (k > High(FDesignedVisible)) or FDesignedVisible[k];   { (as designed: the tool panels hidden) }
        if CardBody(C) <> nil then CardBody(C).Visible := True;
        for i := 0 to C.ControlCount - 1 do
          if C.Controls[i].Align = alBottom then C.Controls[i].Visible := True;
      end;
      MeshingChevron.Caption := #$E2#$96#$BE;
      DisplayChevron.Caption := #$E2#$96#$BE;
      ShapesChevron.Caption := #$E2#$96#$BE;
      B2MChevron.Caption := #$E2#$96#$BE;
      SiamChevron.Caption := #$E2#$96#$BE;
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

{ ------------------------------------------------------------ main menu --- }

{ the main menu as the panels stand: what can run now, the mode, what is shown
  (before each menu opens, and as the buttons change: a short cut acts only
  where its button would) }
procedure TI2MMainForm.SyncMainMenu;
var
  i: Integer;
  Cm: string;
begin
  if MainMenu = nil then Exit;
  MainOpenAny.Enabled := BtnOpen.Enabled;
  MainOpenVolume.Enabled := BtnOpen.Enabled;
  MainOpenMesh.Enabled := BtnOpen.Enabled;
  MainOpenCad.Enabled := BtnOpen.Enabled;
  MainSaveMesh.Enabled := BtnSave.Enabled;
  MainRun.Enabled := BtnRun.Enabled;
  MainStop.Enabled := BtnStop.Enabled;
  MainRunB2M.Enabled := B2MRunBtn.Enabled;
  MainRunSiam.Enabled := SiamRunBtn.Enabled;
  MainPanB2M.Checked := B2MCard.Visible;
  MainPanSiam.Checked := SiamCard.Visible;
  Cm := ChosenMode;
  for i := 0 to MainMode.Count - 1 do
    MainMode.Items[i].Checked := ModeNames[MainMode.Items[i].Tag] = Cm;
  MainShowVol.Checked := ShowVolCheck.Checked;
  MainShowVol.Enabled := ShowVolCheck.Enabled;
  MainShowMesh.Checked := ShowMeshCheck.Checked;
  MainShowMesh.Enabled := ShowMeshCheck.Enabled;
  MainShowEdges.Checked := ShowEdgesCheck.Checked;
  MainShowEdges.Enabled := ShowEdgesCheck.Enabled;
  MainPanMeshing.Checked := MeshingCard.Visible;
  MainPanDisplay.Checked := DisplayCard.Visible;
  MainPanShapes.Checked := ShapesCard.Visible;
  MainMeshPanel.Checked := MeshingCard.Visible;
  MainShapePanel.Checked := ShapesCard.Visible;
  MainShapeDel.Enabled := ShapeDelBtn.Enabled and (Length(FSelPath) > 0);
  MainShapeUp.Enabled := ShapeUpBtn.Enabled and (Length(FSelPath) > 0);
  MainShapeDown.Enabled := ShapeDownBtn.Enabled and (Length(FSelPath) > 0);
  MainShapeSave.Enabled := ShapeSaveBtn.Enabled and FShapesOn;
end;

procedure TI2MMainForm.MainMenuOpen(Sender: TObject);
begin
  SyncMainMenu;
end;

procedure TI2MMainForm.MainExitClick(Sender: TObject);
begin
  Close;
end;

{ Mesh > Make: the Mode section's choice }
procedure TI2MMainForm.MainModeClick(Sender: TObject);
var
  i: Integer;
begin
  for i := 0 to High(Options) do
    if (Options[i].Flag = '--mode') and (FEdits[i] is TComboBox) then
    begin
      TComboBox(FEdits[i]).ItemIndex := TMenuItem(Sender).Tag;
      OptionChanged(FEdits[i]);
    end;
  SyncMainMenu;
end;

{ a settings section by its caption (the item's Hint): its panel shown, it opened }
procedure TI2MMainForm.MainSectionClick(Sender: TObject);
begin
  OpenSection(TMenuItem(Sender).Hint);
end;

{ View > Show ..: the Display panel's check boxes (their OnChange redraws) }
procedure TI2MMainForm.MainDisplayClick(Sender: TObject);
var
  C: TCheckBox;
begin
  case TMenuItem(Sender).Tag of
    1: C := ShowVolCheck;
    2: C := ShowMeshCheck;
  else
    C := ShowEdgesCheck;
  end;
  if C.Enabled then C.Checked := not C.Checked;
  SyncMainMenu;
end;

procedure TI2MMainForm.MainHelpClick(Sender: TObject);
begin
  case TMenuItem(Sender).Tag of
    1: OpenURL(I2MHomePage + '/blob/main/v2m/README.md');
    2: OpenURL(I2MHomePage + '#readme');
  else
    OpenURL(I2MHomePage + '/issues');
  end;
end;

procedure TI2MMainForm.MainAboutClick(Sender: TObject);
begin
  I2MShowAbout(Self, Trim(ExeEdit.Text));
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

{ card k's designed place, kept at its margins to the corner of the view it is
  nearest (as a resize keeps it): a card designed at the top right comes back
  at the top right of a wider window, not where it was in the designed one }
function TI2MMainForm.DefaultRect(k: Integer): TRect;
var
  R: TRect;
  W, H, dx, dy: Integer;
begin
  R := FDefaults[k];
  W := R.Right - R.Left;
  H := R.Bottom - R.Top;
  dx := 0;
  dy := 0;
  if (FDefHost.X > 0) and ((R.Left + R.Right) div 2 > FDefHost.X div 2) then dx := ViewHost.ClientWidth - FDefHost.X;
  if (FDefHost.Y > 0) and ((R.Top + R.Bottom) div 2 > FDefHost.Y div 2) then dy := ViewHost.ClientHeight - FDefHost.Y;
  Result := Rect(Max(0, R.Left + dx), Max(0, R.Top + dy), Max(0, R.Left + dx) + W, Max(0, R.Top + dy) + H);
end;

procedure TI2MMainForm.ApplyLayout(Data: PtrInt);
var
  k: Integer;
begin
  SetLength(FDefaults, Length(AllCards));
  for k := 0 to High(AllCards) do FDefaults[k] := AllCards[k].BoundsRect;
  FDefHost := Point(ViewHost.ClientWidth, ViewHost.ClientHeight);
  SetLength(FDesignedVisible, Length(AllCards));
  for k := 0 to High(AllCards) do FDesignedVisible[k] := AllCards[k].Visible;
  SetLength(FPinned, Length(AllCards));
  SetLength(FAway, Length(AllCards));
  SetLength(FOver, Length(AllCards));
  LoadLayout;
  BuildRecentMenu;
  PictThreshEdit.Enabled := False;
  PictThreshLabel.Enabled := False;
  SetLength(FHome, Length(AllCards));
  for k := 0 to High(AllCards) do
  begin
    SetHome(AllCards[k]);
    PaintPin(AllCards[k]);
    if not FPinned[k] then SetCardCollapsed(AllCards[k], True);   { (auto-hide: its title only) }
  end;
  FHoverTimer := TTimer.Create(Self);
  FHoverTimer.Interval := 120;
  FHoverTimer.OnTimer := @HoverTick;
  FHoverTimer.Enabled := True;
end;

function TI2MMainForm.Pinned(ACard: TPanel): Boolean;
var
  k: Integer;
begin
  k := CardIndex(ACard);
  Result := (k >= 0) and (k <= High(FPinned)) and FPinned[k];
end;

{ the pin, a circle: filled white when pinned (no auto-hide), a grey ring when
  the card auto-hides }
procedure TI2MMainForm.PaintPin(ACard: TPanel);
var
  P: TShape;
begin
  P := TShape(FindComponent(ACard.Name.Replace('Card', 'Pin')));
  if P = nil then Exit;
  if Pinned(ACard) then
  begin
    P.Brush.Style := bsSolid;
    P.Brush.Color := clWhite;
    P.Pen.Color := clWhite;
    P.Hint := 'pinned (no auto-hide); unpin: it hides when the pointer leaves it';
  end
  else
  begin
    P.Brush.Style := bsClear;
    P.Pen.Color := $00A0A0A0;
    P.Hint := 'pin the panel open (no auto-hide); unpinned, it hides when the pointer leaves it';
  end;
end;

procedure TI2MMainForm.CardPinMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var
  C: TPanel;
  k: Integer;
begin
  if Button <> mbLeft then Exit;
  C := CardOf(TControl(Sender));
  k := CardIndex(C);
  if (k < 0) or (k > High(FPinned)) then Exit;
  FPinned[k] := not FPinned[k];
  FAway[k] := 0;
  PaintPin(C);
end;

{ auto-hide: each unpinned card open while the pointer is on it, collapsed
  once the pointer has been off it 0.6 s (not while it is dragged or resized,
  nor while one of its drop-down lists is open) }
procedure TI2MMainForm.HoverTick(Sender: TObject);
const
  HoverOpenMs = 400;
var
  k, i: Integer;
  C: TPanel;
  P, O: TPoint;
  on_, held: Boolean;
  B, U: TControl;
begin
  if not Active then Exit;   { (another window in front: leave the cards as they are) }
  P := Mouse.CursorPos;
  for k := 0 to High(AllCards) do
  begin
    C := AllCards[k];
    if (k > High(FPinned)) or FPinned[k] or not C.Visible then Continue;
    O := C.ClientToScreen(Point(0, 0));
    on_ := (P.X >= O.X - 4) and (P.Y >= O.Y - 4) and (P.X < O.X + C.Width + 4) and (P.Y < O.Y + C.Height + 4);
    B := CardBody(C);
    if on_ then
    begin
      FAway[k] := 0;
      { a collapsed card opens once the pointer rests on it (HoverOpenMs), and
        never while its title is dragged: it can be carried anywhere, the
        bottom edge too, folded }
      if (B <> nil) and not B.Visible then
      begin
        { (the pointer on the card itself, not on a menu, a dialog or another card
          over it: the window under it, through the window system) }
        U := FindControlAtPosition(P, True);
        if (FDragCard = C) or (FSizeCard = C) or (U = nil) or (CardOf(U) <> C) then FOver[k] := 0
        else if FOver[k] = 0 then FOver[k] := GetTickCount64
        else if GetTickCount64 - FOver[k] >= HoverOpenMs then
        begin
          FOver[k] := 0;
          SetCardCollapsed(C, False);
        end;
      end;
      Continue;
    end;
    FOver[k] := 0;
    if (B = nil) or not B.Visible then Continue;   { (collapsed already) }
    held := (FDragCard = C) or (FSizeCard = C);
    for i := 0 to ComponentCount - 1 do
      if (Components[i] is TComboBox) and TComboBox(Components[i]).DroppedDown and (CardOf(TControl(Components[i])) = C) then
        held := True;
    if held then
    begin
      FAway[k] := 0;
      Continue;
    end;
    if FAway[k] = 0 then FAway[k] := GetTickCount64
    else if GetTickCount64 - FAway[k] > 600 then
    begin
      FAway[k] := 0;
      SetCardCollapsed(C, True);
    end;
  end;
end;

function TI2MMainForm.CardIndex(ACard: TPanel): Integer;
var
  k: Integer;
begin
  for k := 0 to High(AllCards) do
    if AllCards[k] = ACard then Exit(k);
  Result := -1;
end;

procedure TI2MMainForm.SetHome(ACard: TPanel);
var
  k: Integer;
begin
  k := CardIndex(ACard);
  { (a card anchored at the bottom: its bottom edge, where its title stays) }
  if (k >= 0) and (k <= High(FHome)) then
    if akBottom in ACard.Anchors then FHome[k] := Point(ACard.Left, ACard.Top + ACard.Height)
    else FHome[k] := Point(ACard.Left, ACard.Top);
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
{ a construct's key without its "(name)" }
function PlainKey(const AKey: string): string;
begin
  Result := AKey;
  if Pos('(', Result) > 0 then Result := Copy(Result, 1, Pos('(', Result) - 1);
end;

{ a boolean operation (its body: the operands), and an object holding one: [root, (Tag)] }
function CsgOp(const AKey: string): Boolean;
begin
  Result := AnsiIndexStr(PlainKey(AKey), ['CSGUnion', 'CSGIntersect', 'CSGSubtract']) >= 0;
end;

function CsgObj(const AKey: string): Boolean;
begin
  Result := PlainKey(AKey) = 'CSGObject';
end;

{ an operand list's bookkeeping member (not a construct) }
function CsgMeta(D: TJSONData): Boolean;
begin
  Result := (D is TJSONObject) and (D.Count >= 1) and
    ((TJSONObject(D).Names[0] = '_DataInfo_') or (TJSONObject(D).Names[0] = 'Tag'));
end;

{ an operand list's operands (its constructs, not its bookkeeping) }
function CsgOperands(A: TJSONArray): Integer;
var
  i: Integer;
begin
  Result := 0;
  for i := 0 to A.Count - 1 do
    if not CsgMeta(A.Items[i]) then Inc(Result);
end;

procedure TI2MMainForm.SetSelPath(const APath: TI2MIntegers);
begin
  FSelPath := Copy(APath);
  if Length(FSelPath) > 0 then FShapeSel := FSelPath[0] else FShapeSel := -1;   { (not IfThen: it reads both) }
end;

{ the construct at APath: its key (a reference: "-> name"), body, the operand list
  holding it (nil: the top level) and its index there }
function TI2MMainForm.ShapeAt(const APath: TI2MIntegers; out AKey: string; out ABody: TJSONData;
  out AParent: TJSONArray; out AIdx: Integer): Boolean;
var
  k: Integer;
  E: TJSONData;
begin
  Result := False;
  AKey := '';
  ABody := nil;
  AParent := nil;
  AIdx := -1;
  if (Length(APath) = 0) or (APath[0] < 0) or (APath[0] >= FShapes.Count) then Exit;
  AKey := FShapes.Key(APath[0]);
  ABody := FShapes.Body(APath[0]);
  AIdx := APath[0];
  for k := 1 to High(APath) do
  begin
    if not (ABody is TJSONArray) or (APath[k] < 0) or (APath[k] >= ABody.Count) then Exit;
    AParent := TJSONArray(ABody);
    AIdx := APath[k];
    E := ABody.Items[APath[k]];
    if (E is TJSONObject) and (E.Count >= 1) then
    begin
      AKey := TJSONObject(E).Names[0];
      ABody := E.Items[0];
    end
    else if (E is TJSONString) and (k = High(APath)) then
    begin
      AKey := '-> ' + E.AsString;
      ABody := E;
    end
    else Exit;
  end;
  Result := True;
end;

{ the tree: each construct, a CSG one's operation and operands under it }
procedure TI2MMainForm.RefreshShapes;

  function TagOf(const AKey: string; D: TJSONData): Integer;   { -1: none }
  var
    j: Integer;
  begin
    Result := -1;
    if (D is TJSONObject) and (TJSONObject(D).Find('Tag') is TJSONNumber) then
      Result := TJSONObject(D).Integers['Tag']
    else if CsgObj(AKey) and (D is TJSONArray) then
      for j := 0 to D.Count - 1 do
        if (D.Items[j] is TJSONObject) and (TJSONObject(D.Items[j]).Find('Tag') is TJSONNumber) then
          Exit(TJSONObject(D.Items[j]).Integers['Tag']);
  end;

  function NewNode(AParent: TTreeNode; const ACaption: string; const APath: TI2MIntegers): TTreeNode;
  begin
    SetLength(FNodePaths, Length(FNodePaths) + 1);
    FNodePaths[High(FNodePaths)] := Copy(APath);
    Result := ShapeTree.Items.AddChild(AParent, ACaption);
    Result.Data := Pointer(PtrInt(High(FNodePaths)));
  end;

  procedure Operands(AParent: TTreeNode; const AKey: string; D: TJSONData; const APath: TI2MIntegers);
  var
    j, n: Integer;
    c, k: string;
    X: TJSONData;
    Nd: TTreeNode;
  begin
    if not (D is TJSONArray) or not (CsgOp(AKey) or CsgObj(AKey)) then Exit;
    n := 0;
    for j := 0 to D.Count - 1 do
    begin
      X := D.Items[j];
      if CsgMeta(X) then Continue;
      if CsgObj(AKey) and (n > 0) then Break;   { (an object: its root alone) }
      if (X is TJSONObject) and (X.Count >= 1) then k := TJSONObject(X).Names[0]
      else if X is TJSONString then k := '-> ' + X.AsString
      else Continue;
      c := k;
      if (PlainKey(AKey) = 'CSGSubtract') and (n > 0) then c := c + '  (taken away)';
      Nd := NewNode(AParent, c, Concat(APath, [j]));
      if X is TJSONObject then Operands(Nd, k, X.Items[0], Concat(APath, [j]));
      Inc(n);
    end;
  end;

  function SamePath(const A, B: TI2MIntegers): Boolean;
  var
    j: Integer;
  begin
    Result := Length(A) = Length(B);
    for j := 0 to High(A) do
      if Result and (A[j] <> B[j]) then Result := False;
  end;

var
  i, t, ix: Integer;
  s: string;
  B: TJSONData;
  Par: TJSONArray;
  N: TTreeNode;
begin
  ShapeTree.Items.BeginUpdate;
  try
    ShapeTree.Items.Clear;
    FNodePaths := nil;
    for i := 0 to FShapes.Count - 1 do
    begin
      B := FShapes.Body(i);
      t := TagOf(FShapes.Key(i), B);
      if CsgObj(FShapes.Key(i)) and (B is TJSONArray) and (B.Count > 0) and (B.Items[0] is TJSONObject) and
        (B.Items[0].Count >= 1) and CsgOp(TJSONObject(B.Items[0]).Names[0]) then
      begin   { a CSG object of an operation: one node, the operation's, its operands under it }
        s := Format('%d. %s', [i + 1, TJSONObject(B.Items[0]).Names[0]]);
        if t >= 0 then s := s + Format('  (Tag %d)', [t]);
        N := NewNode(nil, s, [i]);
        Operands(N, TJSONObject(B.Items[0]).Names[0], B.Items[0].Items[0], [i, 0]);
      end
      else
      begin
        s := Format('%d. %s', [i + 1, FShapes.Key(i)]);
        if t >= 0 then s := s + Format('  (Tag %d)', [t]);
        N := NewNode(nil, s, [i]);
        Operands(N, FShapes.Key(i), B, [i]);
      end;
    end;
    ShapeTree.FullExpand;
  finally
    ShapeTree.Items.EndUpdate;
  end;
  { the selection: its node again (else its construct's, else the last) }
  if (Length(FSelPath) > 0) and (FSelPath[0] >= FShapes.Count) then
    if FShapes.Count > 0 then SetSelPath([FShapes.Count - 1]) else SetSelPath(nil);
  if (Length(FSelPath) > 0) and not ShapeAt(FSelPath, s, B, Par, ix) then SetSelPath([FSelPath[0]]);
  N := nil;
  for i := 0 to ShapeTree.Items.Count - 1 do
    if SamePath(FNodePaths[PtrUInt(ShapeTree.Items[i].Data)], FSelPath) then
    begin
      N := ShapeTree.Items[i];
      Break;
    end;
  if (N = nil) and (Length(FSelPath) = 2) and (FSelPath[1] = 0) then
  begin   { (an object's operation: shown as the object's node) }
    SetSelPath([FSelPath[0]]);
    for i := 0 to ShapeTree.Items.Count - 1 do
      if SamePath(FNodePaths[PtrUInt(ShapeTree.Items[i].Data)], FSelPath) then
      begin
        N := ShapeTree.Items[i];
        Break;
      end;
  end;
  if N <> nil then
  begin
    ShapeTree.OnSelectionChanged := nil;   { (no second refresh) }
    N.Selected := True;
    ShapeTree.OnSelectionChanged := @ShapeTreeChange;
  end;
  FillShapeFields;
  PreviewShapes;
end;

{ the selected construct's members, one row each, as JSON (a Layers row: "Layer i";
  a CSG object: its Tag; an operation: none -- its operands are its children) }
procedure TI2MMainForm.FillShapeFields;
var
  B: TJSONData;
  P: TJSONArray;
  i, ix: Integer;
  k: string;
begin
  ShapeFields.Strings.BeginUpdate;
  try
    ShapeFields.Strings.Clear;
    if not ShapeAt(FSelPath, k, B, P, ix) then Exit;
    if CsgOp(k) then Exit;
    if CsgObj(k) and (B is TJSONArray) then
    begin
      for i := 0 to B.Count - 1 do
        if (B.Items[i] is TJSONObject) and (TJSONObject(B.Items[i]).Find('Tag') <> nil) then
          ShapeFields.Strings.Add('Tag=' + TJSONObject(B.Items[i]).Elements['Tag'].AsJSON);
      if ShapeFields.Strings.Count = 0 then ShapeFields.Strings.Add('Tag=');
    end
    else if B is TJSONString then
      ShapeFields.Strings.Add('reference=' + B.AsJSON)
    else if B is TJSONObject then
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
  if not FShapeFitted or (FShapeGuessed and not S.Guessed) then
  begin   { (fitted once -- and again when a real domain replaces the placeholder one) }
    FShapeFitted := True;
    ResetView;
  end;
  FShapeGuessed := S.Guessed;
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

{ a construct from the Add menu: into the selected boolean operation (or the
  object's, or beside a selected operand) while it has fewer than two operands,
  else a new object at the end; a boolean operation on its own becomes a CSG
  object's (which carries its Tag) }
procedure TI2MMainForm.ShapeAddClick(Sender: TObject);
var
  S: TI2MShapeScene;
  k, sk: string;
  B, SB, R: TJSONData;
  Par, Arr: TJSONArray;
  ix, at: Integer;
  Pre: TI2MIntegers;
  IsOp, IsObj: Boolean;
begin
  k := TMenuItem(Sender).Hint;   { (the key: a caption may gain an accelerator's '&') }
  IsOp := CsgOp(k);
  IsObj := CsgObj(k);
  if not FShapesOn then
  begin   { (no design yet: a new, empty one, this its first construct) }
    FShapes.Clear(False);
    FVolFile := '';
    SetSelPath(nil);
  end;
  S := FShapes.Build(-1);
  if (FShapes.Count = 0) or S.Guessed then
  begin   { (nothing to place it in yet -- no Grid, nothing drawn: MCX Studio's 60-voxel domain) }
    S.Lo.x := 0; S.Lo.y := 0; S.Lo.z := 0;
    S.Hi.x := 60; S.Hi.y := 60; S.Hi.z := 60;
  end;
  { where: the operand list to go in, and at what index }
  Arr := nil;
  at := 0;
  Pre := nil;
  if not IsObj and ShapeAt(FSelPath, sk, SB, Par, ix) then
    if CsgOp(sk) and (SB is TJSONArray) then
    begin
      Arr := TJSONArray(SB);
      at := Arr.Count;
      Pre := Copy(FSelPath);
    end
    else if CsgObj(sk) and (SB is TJSONArray) and (SB.Count > 0) and (SB.Items[0] is TJSONObject) and
      (SB.Items[0].Count >= 1) and CsgOp(TJSONObject(SB.Items[0]).Names[0]) and (SB.Items[0].Items[0] is TJSONArray) then
    begin
      Arr := TJSONArray(SB.Items[0].Items[0]);
      at := Arr.Count;
      Pre := Concat(FSelPath, [0]);
    end
    else if (Par <> nil) and (Length(FSelPath) > 1) then
    begin   { (an operand selected: beside it, in the same operation) }
      Arr := Par;
      at := ix + 1;
      Pre := Copy(FSelPath, 0, Length(FSelPath) - 1);
    end;
  if (Arr <> nil) and (CsgOperands(Arr) >= 2) then
    Arr := nil;   { (a boolean operation takes two operands: a third is a new object) }
  try
    if IsOp then B := TJSONArray.Create
    else if IsObj then B := nil
    else
    begin
      B := GetJSON(I2MDefaultShape(k, S.Lo, S.Hi, FShapes.MaxTag + 1));
      if (Arr <> nil) and (B is TJSONObject) and (TJSONObject(B).IndexOfName('Tag') >= 0) then
        TJSONObject(B).Delete('Tag');   { (an operand: the object's Tag labels it) }
    end;
  except
    on E: Exception do
    begin
      ShapeHint.Caption := 'could not add ' + k + ': ' + E.Message;
      Exit;
    end;
  end;
  if Arr <> nil then
  begin
    Arr.Insert(at, TJSONObject.Create([k, B]));
    SetSelPath(Concat(Pre, [at]));
  end
  else if IsOp or IsObj then
  begin   { a new CSG object: [the operation, a Tag member], its operation selected (shapes go in) }
    if IsObj then k := 'CSGUnion';
    if B = nil then B := TJSONArray.Create;
    R := TJSONArray.Create([TJSONObject.Create([k, B]), TJSONObject.Create(['Tag', FShapes.MaxTag + 1])]);
    FShapes.Add('CSGObject', R);
    SetSelPath([FShapes.Count - 1]);   { (the object and its operation: one node) }
  end
  else
  begin
    FShapes.Add(k, B);
    SetSelPath([FShapes.Count - 1]);
  end;
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeDelClick(Sender: TObject);
var
  k: string;
  B: TJSONData;
  Par: TJSONArray;
  ix: Integer;
begin
  if not ShapeAt(FSelPath, k, B, Par, ix) then Exit;
  if Par = nil then
  begin
    FShapes.Delete(ix);
    if ix >= FShapes.Count then ix := FShapes.Count - 1;
    if ix >= 0 then SetSelPath([ix]) else SetSelPath(nil);
  end
  else
  begin   { an operand: out of its operation, which is then selected }
    Par.Delete(ix);
    SetSelPath(Copy(FSelPath, 0, Length(FSelPath) - 1));
  end;
  ShapesChanged;
end;

{ up / down: among the top-level constructs, or an operand among its operation's }
procedure TI2MMainForm.ShapeMove(ADelta: Integer);
var
  k: string;
  B: TJSONData;
  Par: TJSONArray;
  ix, j: Integer;
  P: TI2MIntegers;
begin
  if not ShapeAt(FSelPath, k, B, Par, ix) then Exit;
  j := ix + ADelta;
  if Par = nil then
  begin
    if (j < 0) or (j >= FShapes.Count) then Exit;
    FShapes.Move(ix, j);
    SetSelPath([j]);
  end
  else
  begin
    if (j < 0) or (j >= Par.Count) or CsgMeta(Par.Items[j]) or CsgMeta(Par.Items[ix]) then Exit;
    Par.Exchange(ix, j);
    P := Copy(FSelPath);
    P[High(P)] := j;
    SetSelPath(P);
  end;
  ShapesChanged;
end;

procedure TI2MMainForm.ShapeUpClick(Sender: TObject);
begin
  ShapeMove(-1);
end;

procedure TI2MMainForm.ShapeDownClick(Sender: TObject);
begin
  ShapeMove(1);
end;

procedure TI2MMainForm.ShapeNewClick(Sender: TObject);
begin
  if FShapesOn and FShapesDirty and (MessageDlg('New design', 'Discard the changes to the current design?',
     mtConfirmation, [mbYes, mbNo], 0) <> mrYes) then Exit;
  FShapes.Clear;
  FVolFile := '';
  SetSelPath([0]);
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
    if D.Execute and LoadImage(D.FileName) then AddRecent(D.FileName);
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
  SetSelPath(FNodePaths[PtrUInt(ShapeTree.Selected.Data)]);
  FillShapeFields;
  PreviewShapes;
end;

{ a value edited: parsed as JSON (else taken as a string), into the construct }
procedure TI2MMainForm.ShapeFieldValidate(Sender: TObject; ACol, ARow: Integer; const OldValue: string;
  var NewValue: string);
var
  B, D: TJSONData;
  Par: TJSONArray;
  Name_, k: string;
  i, ix: Integer;
  Done: Boolean;
begin
  if (NewValue = OldValue) or (ARow < 1) or not ShapeAt(FSelPath, k, B, Par, ix) then Exit;
  try
    D := GetJSON(NewValue);
  except
    D := TJSONString.Create(NewValue);   { (bare text: a string) }
  end;
  Name_ := ShapeFields.Keys[ARow];
  if CsgObj(k) and (B is TJSONArray) then
  begin   { the object's Tag: in its Tag member }
    Done := False;
    for i := 0 to B.Count - 1 do
      if (B.Items[i] is TJSONObject) and (TJSONObject(B.Items[i]).IndexOfName('Tag') >= 0) then
      begin
        TJSONObject(B.Items[i]).Elements['Tag'] := D;
        Done := True;
        Break;
      end;
    if not Done then TJSONArray(B).Add(TJSONObject.Create(['Tag', D]));
  end
  else if (B is TJSONString) and (Par <> nil) then
  begin   { a reference: another name }
    Par.Items[ix] := TJSONString.Create(D.AsString);
    D.Free;
    NewValue := Par.Items[ix].AsJSON;
    FShapesDirty := True;
    ShapesChanged;
    Exit;
  end
  else if B is TJSONObject then
  begin
    i := TJSONObject(B).IndexOfName(Name_);
    if i >= 0 then TJSONObject(B).Items[i] := D else TJSONObject(B).Add(Name_, D);
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
  k: Integer;
begin
  if not FileExists(LayoutFile) then Exit;
  Ini := TIniFile.Create(LayoutFile);
  try
    FLastDir := Ini.ReadString('Files', 'LastDir', '');
    FGrayMap := EnsureRange(Ini.ReadInteger('View', 'ColourMap', 0), 0, MapCombo.Items.Count - 1);
    FRecent.Clear;
    for k := 1 to MaxRecent do
      if Ini.ReadString('Recent', 'File' + IntToStr(k), '') <> '' then
        FRecent.Add(Ini.ReadString('Recent', 'File' + IntToStr(k), ''));
    { (the programs last used, where they still are) }
    if FileExists(Ini.ReadString('Tools', 'brain2mesh', '')) then B2MExeEdit.Text := Ini.ReadString('Tools', 'brain2mesh', '');
    if FileExists(Ini.ReadString('Tools', 'siamize', '')) then SiamExeEdit.Text := Ini.ReadString('Tools', 'siamize', '');
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
      if CardIndex(C) <= High(FPinned) then FPinned[CardIndex(C)] := Ini.ReadBool(C.Name, 'Pinned', False);
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
      Ini.WriteInteger('View', 'ColourMap', FGrayMap);
      if FileExists(B2MExeEdit.Text) then Ini.WriteString('Tools', 'brain2mesh', B2MExeEdit.Text);
      if FileExists(SiamExeEdit.Text) then Ini.WriteString('Tools', 'siamize', SiamExeEdit.Text);
      for C in AllCards do
      begin
        Collapsed := (CardBody(C) <> nil) and not CardBody(C).Visible;
        Ini.WriteInteger(C.Name, 'Left', ScaleScreenTo96(C.Left));
        Ini.WriteInteger(C.Name, 'Top', ScaleScreenTo96(C.Top));
        Ini.WriteInteger(C.Name, 'Width', ScaleScreenTo96(C.Width));
        Ini.WriteInteger(C.Name, 'Height', ScaleScreenTo96(C.Height));
        Ini.WriteBool(C.Name, 'Visible', C.Visible);
        Ini.WriteBool(C.Name, 'Collapsed', Collapsed);
        Ini.WriteBool(C.Name, 'Pinned', Pinned(C));
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
