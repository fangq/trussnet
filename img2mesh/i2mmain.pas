{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mmain -- the window: open an image, set trussnet's options, run it, look at
  the image and the mesh together, cropped and translucent.

  The form is built in code from a table of trussnet's options (Options below),
  so a new trussnet flag is one line here. An empty field is trussnet's own
  default, which is shown greyed in the field. }
unit i2mmain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, StrUtils, Process, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, ComCtrls, CheckLst, Buttons, LCLType, LCLIntf, ImgList, ToolWin, mcxgl, i2mvol, i2mmesh, i2mview, i2micons, i2maccord;

type
  TI2MOptKind = (okFloat, okInt, okText, okBool, okChoice, okFlagArg);

  TI2MOption = record
    Flag: string;      { the trussnet flag }
    Caption: string;
    Kind: TI2MOptKind;
    Default: string;   { shown greyed; for a choice, the items, '|'-separated }
    Hint: string;
    Group: string;     { a heading starts a new group }
  end;

  TI2MMainForm = class(TForm)
  private
    { layout }
    FTool: TToolBar;
    FIcons: TImageList;
    FNav: TI2MAccordion;
    FViewHost: TPanel;
    FBottom: TPanel;
    FLog: TMemo;
    FCmd: TEdit;
    FStatus: TStatusBar;
    FBtnOpen, FBtnRun, FBtnStop, FBtnMesh, FBtnSave, FBtnFit, FBtnShot: TToolButton;
    { meshing options }
    FExe: TEdit;
    FFormat: TComboBox;
    FExtra: TEdit;
    FEdits: array of TControl;   { per option: TEdit / TCheckBox / TComboBox }
    FArgEdits: array of TEdit;   { okFlagArg: the argument }
    { display }
    FShowVol, FShowMesh, FShowEdges: TCheckBox;
    FMap, FStyle, FChannel: TComboBox;
    FOpacity, FFloor, FMeshAlpha: TTrackBar;
    FLabels: TCheckListBox;
    FClip: array[0..5] of TTrackBar;
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
    procedure BuildLayout;
    procedure BuildMeshingSections;
    procedure BuildDisplaySections;
    function AddLabel(AParent: TWinControl; const ACaption: string; ABold: Boolean): TLabel;
    procedure Log(const AText: string);
    procedure ViewLog(Sender: TObject; const AText: string);
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
    procedure FitClick(Sender: TObject);
    procedure ShotClick(Sender: TObject);
    procedure ResetClipClick(Sender: TObject);
    procedure BrowseExeClick(Sender: TObject);
    procedure Poll(Sender: TObject);
    procedure Drain;
    procedure Finished;
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormDropFiles(Sender: TObject; const FileNames: array of string);
    function Arguments(out AList: TStringList): Boolean;
    procedure UpdateCommand;
    procedure UpdateButtons;
    procedure ShowChannel;
    procedure FillLabels;
    procedure UploadVolume;
    procedure ScanVolumeLabels;
    procedure AllLabelsClick(Sender: TObject);
    function ChannelName(AChannel: Integer): string;
    function OptionText(const AFlag: string): string;
    function ArgmaxKey: string;
    function VoxelSize: TMcxVec3;
    procedure ClipTimer(Sender: TObject);
    function FindTrussnet: string;
    { --mode, from the Make choice; its input is the mesh shown (not the image) }
    function Mode: string;
    function MeshInput: Boolean;
    function InputMeshFile: string;
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
    procedure FitView;
    { opens a section of the left panel: 0 the first meshing one, 1 the crop box }
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

const
  Options: array[0..39] of TI2MOption = (
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
     Hint: 'remesh / repair: the spacing of the fields the surfaces are rasterized into'; Group: ''),
    (Flag: '--cdt-fill'; Caption: 'CDT interior spacing'; Kind: okFloat; Default: 'size (0 = none)';
     Hint: 'cdt: the spacing of the interior points (default: the size, else 1.5 x the mean edge)'; Group: ''),
    (Flag: '--opt-rounds'; Caption: 'Optimiser rounds'; Kind: okInt; Default: '3';
     Hint: 'cdt / optimize: rounds of flips, collapses, Steiner points and smoothing'; Group: ''),
    (Flag: '--overlap'; Caption: 'Overlap rule'; Kind: okChoice; Default: '(default: nest)|split|max|min|union|cells';
     Hint: 'remesh / repair, surfaces that cross: who owns a volume two regions claim -- nest: the smaller, ' +
       'split: halfway, max / min: the label, union: one region, cells: each overlap a region'; Group: ''),
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

  ClipNames: array[0..5] of string = ('x from', 'x to', 'y from', 'y to', 'z from', 'z to');
  ClipSteps = 200;

function P3(x, y, z: Single): TI2MPoint;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

{ ------------------------------------------------------------- building --- }

constructor TI2MMainForm.Create(AOwner: TComponent);
begin
  inherited CreateNew(AOwner);
  Caption := 'img2mesh - trussnet';
  Width := 1360;
  Height := 860;
  Position := poScreenCenter;
  AllowDropFiles := True;
  OnDropFiles := @FormDropFiles;
  OnClose := @FormClose;
  BuildLayout;
  { the first field would take the focus and hide its greyed default }
  ActiveControl := FNav;
  FView := TI2MView.Create(FViewHost, I2MGLMode);
  FView.OnLog := @ViewLog;
  Log('display: ' + FView.Backend);
  FTimer := TTimer.Create(Self);
  FTimer.Enabled := False;
  FTimer.Interval := 100;
  FTimer.OnTimer := @Poll;
  FClipTimer := TTimer.Create(Self);
  FClipTimer.Enabled := False;
  FClipTimer.Interval := 60;
  FClipTimer.OnTimer := @ClipTimer;
  FExe.Text := FindTrussnet;
  DisplayChanged(nil);
  UpdateCommand;
  UpdateButtons;
end;

destructor TI2MMainForm.Destroy;
begin
  if Running then FProc.Terminate(1);
  FreeAndNil(FProc);
  FreeAndNil(FView);
  FreeAndNil(FMesh);
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

function TI2MMainForm.AddLabel(AParent: TWinControl; const ACaption: string; ABold: Boolean): TLabel;
begin
  Result := TLabel.Create(Self);
  Result.Parent := AParent;
  Result.Caption := ACaption;
  Result.Align := alTop;
  Result.Top := Next;
  Result.BorderSpacing.Top := IfThen(ABold, 10, 4);
  Result.BorderSpacing.Left := 4;
  if ABold then Result.Font.Style := [fsBold];
end;

procedure TI2MMainForm.BuildLayout;

  { a toolbar button: MCX Studio's icon over its caption }
  function Btn(const AIcon, ACaption, AHint: string; AClick: TNotifyEvent): TToolButton;
  var
    G: TBitmap;
  begin
    Result := TToolButton.Create(FTool);
    Result.Parent := FTool;
    Result.Left := Next * 100;   { after the ones before it }
    Result.Caption := ACaption;
    Result.Hint := AHint;
    Result.OnClick := AClick;
    G := I2MIcon(AIcon, FIcons.Width);
    if G <> nil then
    begin
      Result.ImageIndex := FIcons.Add(G, nil);
      G.Free;
    end;
  end;

  procedure Divider;
  var
    D: TToolButton;
  begin
    D := TToolButton.Create(FTool);
    D.Parent := FTool;
    D.Left := Next * 100;
    D.Style := tbsDivider;
  end;

var
  Split: TSplitter;
  sz: Integer;
begin
  sz := MulDiv(40, Screen.PixelsPerInch, 96);
  FIcons := TImageList.Create(Self);
  FIcons.Width := sz;
  FIcons.Height := sz;
  FTool := TToolBar.Create(Self);
  FTool.Parent := Self;
  FTool.Align := alTop;
  FTool.Images := FIcons;
  FTool.ShowCaptions := True;
  FTool.ButtonWidth := MulDiv(92, Screen.PixelsPerInch, 96);
  FTool.ButtonHeight := sz + MulDiv(30, Screen.PixelsPerInch, 96);
  FTool.AutoSize := True;
  FTool.Flat := True;
  FTool.EdgeBorders := [ebBottom];
  FTool.ShowHint := True;
  FTool.Indent := MulDiv(6, Screen.PixelsPerInch, 96);
  FBtnOpen := Btn('open', 'Open image', 'a label, gray-scale or 4-D probability image: .nii .nii.gz .jnii .bnii', @OpenClick);
  FBtnMesh := Btn('tetmesh', 'Open mesh', 'show a mesh or a surface (.jmsh .bmsh .off .stl); the input of the ' +
    'remesh / repair / cdt / optimize modes', @MeshClick);
  Divider;
  FBtnRun := Btn('run', 'Run', 'mesh the image with trussnet, with the settings on the left', @RunClick);
  FBtnStop := Btn('stop', 'Stop', 'stop the running trussnet', @StopClick);
  Divider;
  FBtnSave := Btn('saveas', 'Save mesh', 'keep the mesh trussnet made', @SaveClick);
  FBtnShot := Btn('save', 'Save picture', 'the view as a PNG', @ShotClick);
  FBtnFit := Btn('fit', 'Fit view', 'frame the image / mesh', @FitClick);

  FStatus := TStatusBar.Create(Self);
  FStatus.Parent := Self;
  FStatus.SimplePanel := True;
  FStatus.SimpleText := 'Open an image (or drop one on the window) to start.';

  FNav := TI2MAccordion.Create(Self);
  FNav.Parent := Self;
  FNav.Align := alLeft;
  FNav.Width := MulDiv(300, Screen.PixelsPerInch, 96);
  BuildMeshingSections;
  BuildDisplaySections;
  FNav.Open(0);

  Split := TSplitter.Create(Self);
  Split.Parent := Self;
  Split.Align := alLeft;
  Split.Left := FNav.Width + 1;

  FBottom := TPanel.Create(Self);
  FBottom.Parent := Self;
  FBottom.Align := alBottom;
  FBottom.Height := 170;
  FBottom.BevelOuter := bvNone;

  FCmd := TEdit.Create(Self);
  FCmd.Parent := FBottom;
  FCmd.Align := alTop;
  FCmd.ReadOnly := True;
  FCmd.Font.Name := 'Monospace';
  FCmd.Hint := 'the command Run will start';
  FCmd.ShowHint := True;

  FLog := TMemo.Create(Self);
  FLog.Parent := FBottom;
  FLog.Align := alClient;
  FLog.ReadOnly := True;
  FLog.ScrollBars := ssAutoBoth;
  FLog.WordWrap := False;
  FLog.Font.Name := 'Monospace';

  Split := TSplitter.Create(Self);
  Split.Parent := Self;
  Split.Align := alBottom;
  Split.Top := FBottom.Top - 1;

  FBottom.Top := Next;
  Split.Top := FBottom.Top - 5;
  FStatus.Top := Next + 1000;

  FViewHost := TPanel.Create(Self);
  FViewHost.Parent := Self;
  FViewHost.Align := alClient;
  FViewHost.BevelOuter := bvNone;
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
  B: TButton;
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
    Box := FNav.AddSection(ACaption);
  end;

  procedure ProgramSection;
  begin
  Heading('TrussNet path');
  Row := NewRow;
  RowLabel(Row, 'Executable');
  B := TButton.Create(Self);
  B.Parent := Row;
  B.Align := alRight;
  B.Caption := '...';
  B.Width := 30;
  B.OnClick := @BrowseExeClick;
  FExe := TEdit.Create(Self);
  FExe.Parent := Row;
  FExe.Align := alClient;
  FExe.BorderSpacing.Right := 2;
  FExe.OnChange := @OptionChanged;
  Row := NewRow;
  RowLabel(Row, 'Output format');
  FFormat := TComboBox.Create(Self);
  FFormat.Parent := Row;
  FFormat.Align := alClient;
  FFormat.Style := csDropDownList;
  FFormat.Items.Add('.jmsh (JSON text)');
  FFormat.Items.Add('.bmsh (binary JSON)');
  FFormat.ItemIndex := 1;
  FFormat.BorderSpacing.Right := 4;
  FFormat.OnChange := @OptionChanged;
  end;

begin
  Box := nil;
  Lefts := nil;
  FNav.AddGroup('Meshing', 0);
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

  ProgramSection;
  Heading('Other arguments');
  Row := NewRow;
  FExtra := TEdit.Create(Self);
  FExtra.Parent := Row;
  FExtra.Align := alClient;
  FExtra.TextHint := 'passed as they are, e.g. --fscale 1.1';
  FExtra.BorderSpacing.Left := 6;
  FExtra.BorderSpacing.Right := 4;
  FExtra.OnChange := @OptionChanged;

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

procedure TI2MMainForm.BuildDisplaySections;
var
  Box: TPanel;   { the current section's body }
  i: Integer;
  B: TBitBtn;
  G: TBitmap;
  Row: TPanel;

  function Check(const ACaption: string): TCheckBox;
  begin
    Result := TCheckBox.Create(Self);
    Result.Parent := Box;
    Result.Align := alTop;
    Result.Top := Next;
    Result.Caption := ACaption;
    Result.Checked := True;
    Result.BorderSpacing.Left := 6;
    Result.OnChange := @DisplayChanged;
  end;

  function Combo(const ACaption, AItems: string; AIndex: Integer): TComboBox;
  var
    s: string;
  begin
    AddLabel(Box, ACaption, False);
    Result := TComboBox.Create(Self);
    Result.Parent := Box;
    Result.Align := alTop;
    Result.Top := Next;
    Result.Style := csDropDownList;
    for s in AItems.Split('|') do Result.Items.Add(s);
    Result.ItemIndex := AIndex;
    Result.BorderSpacing.Left := 6;
    Result.BorderSpacing.Right := 6;
    Result.OnChange := @DisplayChanged;
  end;

  function Track(const ACaption: string; AMax, APos: Integer): TTrackBar;
  begin
    AddLabel(Box, ACaption, False);
    Result := TTrackBar.Create(Self);
    Result.Parent := Box;
    Result.Align := alTop;
    Result.Top := Next;
    Result.Max := AMax;
    Result.Position := APos;
    Result.TickStyle := tsNone;
    Result.Height := 26;
    Result.BorderSpacing.Left := 4;
    Result.BorderSpacing.Right := 4;
    Result.OnChange := @DisplayChanged;
  end;

begin
  FNav.AddGroup('Display', 1);
  Box := FNav.AddSection('Crop box');
  for i := 0 to 5 do
  begin
    FClip[i] := Track(ClipNames[i], ClipSteps, IfThen(Odd(i), ClipSteps, 0));
    FClip[i].OnChange := @ClipChanged;
  end;
  B := TBitBtn.Create(Self);
  G := I2MIcon('reset', I2MIconSize);
  if G <> nil then
  begin
    B.Glyph.Assign(G);
    G.Free;
  end;
  B.Parent := Box;
  B.Align := alTop;
  B.Top := Next;
  B.Caption := 'Reset the crop box';
  B.BorderSpacing.Around := 6;
  B.OnClick := @ResetClipClick;

  Box := FNav.AddSection('Labels');
  FLabels := TCheckListBox.Create(Self);
  FLabels.Parent := Box;
  FLabels.Align := alTop;
  FLabels.Top := Next;
  FLabels.Height := 130;
  FLabels.BorderSpacing.Left := 6;
  FLabels.BorderSpacing.Right := 6;
  FLabels.OnClickCheck := @LabelsChanged;
  FLabels.Hint := 'untick a label to hide it in the mesh and the image';
  FLabels.ShowHint := True;
  Row := TPanel.Create(Self);
  Row.Parent := Box;
  Row.Align := alTop;
  Row.Top := Next;
  Row.Height := 32;
  Row.BevelOuter := bvNone;
  for i := 0 to 1 do
  begin
    B := TBitBtn.Create(Self);
    B.Parent := Row;
    B.Align := alLeft;
    B.Left := Next;
    B.AutoSize := True;
    B.BorderSpacing.Around := 3;
    if i = 0 then B.Caption := 'Show all' else B.Caption := 'Hide all';
    B.Tag := i;
    B.OnClick := @AllLabelsClick;
  end;

  Box := FNav.AddSection('Image');
  FShowVol := Check('Show the image');
  FChannel := Combo('Channel (4-D)', 'argmax', 0);
  FChannel.OnChange := @ChannelChanged;
  FMap := Combo('Colour map', 'jet|hot|viridis|cool|grey', 0);
  FStyle := Combo('Rendering', 'maximum intensity|accumulate', 1);
  FOpacity := Track('Opacity', 100, 30);
  FFloor := Track('Hide below (fraction of range)', 100, 2);

  Box := FNav.AddSection('Mesh');
  FShowMesh := Check('Show the mesh');
  FShowEdges := Check('Show the edges');
  FMeshAlpha := Track('Surface opacity', 100, 100);

end;

{ -------------------------------------------------------------- logging --- }

procedure TI2MMainForm.Log(const AText: string);
begin
  FLog.Lines.Add(AText);
  FLog.SelStart := Length(FLog.Text);
  if FEcho then WriteLn(AText);
end;

procedure TI2MMainForm.ViewLog(Sender: TObject; const AText: string);
begin
  Log(AText);
end;

{ -------------------------------------------------------------- options --- }

function TI2MMainForm.FindTrussnet: string;
var
  Here, BinName, c: string;
  Cands: array of string;
begin
  {$IFDEF WINDOWS}BinName := 'trussnet.exe';{$ELSE}BinName := 'trussnet';{$ENDIF}
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
    Result := FVolFile <> '';
    if FVolFile <> '' then AList.Add(FVolFile) else AList.Add('<image>');
  end;
  if FFormat.ItemIndex = 0 then v := '.jmsh' else v := '.bmsh';
  if FOutFile = '' then
    FOutFile := IncludeTrailingPathDelimiter(GetTempDir(False)) +
      Format('img2mesh-%d', [GetProcessID]) + v
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
  if Trim(FExtra.Text) <> '' then
  begin
    Extra := TStringList.Create;
    try
      CommandToList(Trim(FExtra.Text), Extra);
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
  if FCmd = nil then Exit;
  Arguments(L);
  try
    s := FExe.Text;
    for a in L do
      if (Pos(' ', a) > 0) or (a = '') then s := s + ' "' + a + '"' else s := s + ' ' + a;
    FCmd.Text := s;
  finally
    L.Free;
  end;
end;

procedure TI2MMainForm.OptionChanged(Sender: TObject);
begin
  UpdateCommand;
  UpdateButtons;
  { the argmax view follows trussnet's channel options }
  if (FVol.Nc > 1) and (FChannel.ItemIndex = 0) and (ArgmaxKey <> FArgmaxKey) then
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
  FExtra.Text := Trim(FExtra.Text + ' ' + AFlag + ' ' + AValue);
end;

procedure TI2MMainForm.BrowseExeClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  try
    D.Title := 'The trussnet executable';
    D.FileName := FExe.Text;
    if D.Execute then FExe.Text := D.FileName;
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.UpdateButtons;
begin
  if FBtnRun = nil then Exit;
  if MeshInput then FBtnRun.Enabled := (FMeshFile <> '') and not Running
  else FBtnRun.Enabled := (FVolFile <> '') and not Running;
  FBtnStop.Enabled := Running;
  FBtnSave.Enabled := (FMesh <> nil) and (FMeshFile <> '');
  FBtnOpen.Enabled := not Running;
end;

{ -------------------------------------------------------------- display --- }

procedure TI2MMainForm.DisplayChanged(Sender: TObject);
begin
  if (FView = nil) or FUpdating then Exit;
  FView.ShowVolume := FShowVol.Checked;
  FView.ShowMesh := FShowMesh.Checked;
  FView.ShowEdges := FShowEdges.Checked;
  FView.Colormap := FMap.ItemIndex;
  FView.Style := FStyle.ItemIndex;
  FView.Opacity := FOpacity.Position / 100;
  FView.Threshold := FFloor.Position / 100;
  FView.MeshAlpha := FMeshAlpha.Position / 100;   { a uniform: no rebuild }
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
  FLabels.Items.BeginUpdate;
  try
    FLabels.Items.Clear;
    for t := 0 to High(Seen) do
      if Seen[t] then
      begin
        n := 'label ' + IntToStr(t);
        if (t <= High(FLabelNames)) and (FLabelNames[t] <> '') then n := n + ' (' + FLabelNames[t] + ')';
        k := FLabels.Items.AddObject(n, TObject(PtrInt(t)));
        FLabels.Checked[k] := FView.LabelVisible[t];
      end;
  finally
    FLabels.Items.EndUpdate;
  end;
end;

procedure TI2MMainForm.LabelsChanged(Sender: TObject);
var
  k: Integer;
begin
  for k := 0 to FLabels.Items.Count - 1 do
    FView.LabelVisible[PtrInt(FLabels.Items.Objects[k])] := FLabels.Checked[k];
  FView.MeshChanged;
  UploadVolume;
end;

procedure TI2MMainForm.HideLabels(const AList: string);
var
  k, t: Integer;
  it: string;
begin
  for it in AList.Split([','], TStringSplitOptions.ExcludeEmpty) do
    if TryStrToInt(Trim(it), t) then
      for k := 0 to FLabels.Items.Count - 1 do
        if PtrInt(FLabels.Items.Objects[k]) = t then FLabels.Checked[k] := False;
  LabelsChanged(nil);
end;

procedure TI2MMainForm.AllLabelsClick(Sender: TObject);
var
  k: Integer;
begin
  for k := 0 to FLabels.Items.Count - 1 do
    FLabels.Checked[k] := TComponent(Sender).Tag = 0;
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
  c := FChannel.ItemIndex - 1;
  if (FVol.Nc > 1) and (c < 0) then
  begin   { trussnet's labels: its exterior channels, maps and thresholds }
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
  Result := I2MLoadVolume(AFileName, FVol, Err);
  if not Result then
  begin
    Log('could not read ' + AFileName + ': ' + Err);
    FStatus.SimpleText := 'could not read ' + ExtractFileName(AFileName);
    Exit;
  end;
  FVolFile := ExpandFileName(AFileName);
  Log(Format('%s: %d x %d x %d%s, %s, %.4g .. %.4g, voxel %.3g x %.3g x %.3g mm (%d ms)',
    [ExtractFileName(AFileName), FVol.Nx, FVol.Ny, FVol.Nz,
     IfThen(FVol.Nc > 1, Format(' x %d channels', [FVol.Nc]), ''),
     IfThen(FVol.IsInteger, 'integer (labels)', 'real'), FVol.Low, FVol.High,
     FVol.VoxelSize[0], FVol.VoxelSize[1], FVol.VoxelSize[2], GetTickCount64 - T0]));
  FChannel.Items.Clear;
  if FVol.Nc > 1 then
  begin
    FChannel.Items.Add('argmax (labels)');
    for c := 0 to FVol.Nc - 1 do FChannel.Items.Add('channel ' + ChannelName(c));
    FChannel.ItemIndex := 0;
    FChannel.Enabled := True;
  end
  else
  begin
    FChannel.Items.Add('(one channel)');
    FChannel.ItemIndex := 0;
    FChannel.Enabled := False;
  end;
  { a label image: the exterior (0) hidden, one colour per label }
  FUpdating := True;
  if FVol.IsInteger or (FVol.Nc > 1) then
  begin
    FMap.ItemIndex := 0;
    FFloor.Position := 1;
  end
  else
  begin
    FMap.ItemIndex := 4;
    FFloor.Position := 5;
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
  Caption := 'img2mesh - ' + ExtractFileName(AFileName);
  FStatus.SimpleText := Format('%s: %d x %d x %d%s', [ExtractFileName(AFileName),
    FVol.Nx, FVol.Ny, FVol.Nz, IfThen(FVol.Nc > 1, Format(' x %d', [FVol.Nc]), '')]);
  ResetBox;
  UpdateCommand;
  UpdateButtons;
end;

procedure TI2MMainForm.ResetBox;
begin
  SetClip(McxVec3(0, 0, 0), McxVec3(1, 1, 1));
  FView.FitView;
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
  { trussnet writes world = affine x (i, j, k); the texture has voxel i's
    centre at i + 0.5 }
  if FVol.Nx > 0 then
    with VoxelSize do M.ToDisplay(FVol.Affine, 0.5, P3(x, y, z));
  First := FMesh = nil;
  FView.SetMesh(nil);
  FreeAndNil(FMesh);
  FMesh := M;
  FMeshFile := AFileName;
  { a dense mesh's wireframe is a solid colour at any ordinary zoom }
  if First then
  begin
    ShowOnly('mesh');   { the image drawn over it hides it; one click brings it back }
  end;
  FView.SetMesh(FMesh);
  FillLabels;
  Log(Format('%s: %d nodes, %d tets, %d surface triangles (%d ms)',
    [ExtractFileName(AFileName), M.NodeCount, M.ElemCount, M.FaceCount, GetTickCount64 - T0]));
  if M.IsSurface then
    FStatus.SimpleText := Format('surface: %d nodes, %d triangles', [M.NodeCount, M.FaceCount])
  else
    FStatus.SimpleText := Format('mesh: %d nodes, %d tets', [M.NodeCount, M.ElemCount]);
  if AReset then ResetBox
  else if First and (FVol.Nx = 0) then FView.FitView;
  UpdateButtons;
end;

procedure TI2MMainForm.OpenClick(Sender: TObject);
var
  D: TOpenDialog;
begin
  D := TOpenDialog.Create(Self);
  try
    D.Title := 'Open an image';
    D.Filter := 'Images (*.nii;*.nii.gz;*.jnii;*.bnii)|*.nii;*.nii.gz;*.gz;*.jnii;*.bnii|All files|*';
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
  finally
    D.Free;
  end;
end;

procedure TI2MMainForm.FitClick(Sender: TObject);
begin
  FitView;
end;

procedure TI2MMainForm.ShowOnly(const AWhat: string);
begin
  FShowVol.Checked := AWhat <> 'mesh';
  FShowMesh.Checked := AWhat <> 'volume';
  DisplayChanged(nil);
end;

procedure TI2MMainForm.ShowPage(AIndex: Integer);
begin
  if AIndex = 1 then OpenSection('Crop box') else FNav.Open(0);
end;

procedure TI2MMainForm.OpenSection(const ACaption: string);
begin
  FNav.Open(FNav.IndexOf(ACaption));
end;

procedure TI2MMainForm.FitView;
begin
  FView.FitView;
end;

procedure TI2MMainForm.ShotClick(Sender: TObject);
var
  D: TSaveDialog;
begin
  D := TSaveDialog.Create(Self);
  try
    D.Title := 'Save the view as';
    D.DefaultExt := 'png';
    D.Filter := 'PNG (*.png)|*.png';
    D.Options := D.Options + [ofOverwritePrompt];
    if D.Execute then
      if SaveImage(D.FileName, FViewHost.Width, FViewHost.Height) then Log('saved ' + D.FileName);
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
     n.EndsWith('.bnii') then Exit(1);
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
          Log('trussnet is running; not opening ' + ExtractFileName(f));
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
  FProc.Executable := FExe.Text;
  FProc.Parameters.Assign(L);
  L.Free;
  FProc.Options := [poUsePipes, poStderrToOutPut, poNoConsole];
  FPending := '';
  Log('');
  Log('$ ' + FCmd.Text);
  try
    FProc.Execute;
  except
    on E: Exception do
    begin
      Log('could not start ' + FExe.Text + ': ' + E.Message);
      FreeAndNil(FProc);
      UpdateButtons;
      Exit;
    end;
  end;
  FStarted := Now;
  FTimer.Enabled := True;
  FStatus.SimpleText := 'trussnet is running...';
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
    FStatus.SimpleText := Format('trussnet is running... %.1f s', [(Now - FStarted) * 86400]);
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
  Log(Format('trussnet finished in %.1f s, exit code %d', [(Now - FStarted) * 86400, Code]));
  if (Code = 0) and FileExists(FOutFile) then
  begin
    LoadMesh(FOutFile, False);
    ShowOnly('mesh');
  end
  else
    FStatus.SimpleText := Format('trussnet failed (exit code %d); see the log', [Code]);
  UpdateButtons;
end;

procedure TI2MMainForm.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if Running then FProc.Terminate(1);
  CloseAction := caFree;
end;

end.
