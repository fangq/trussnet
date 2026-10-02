{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mview -- the 3-D view: an image as a raycast volume, a v2mesh mesh as a
  cut-out of its tetrahedra, both cropped to an x/y/z box.

  The drawing is mcxstudio2's (mcxgl: the camera, the shaders, the volume
  raycaster; same author, GPL-3.0-or-later). The mesh is the surface of the
  tets inside the box (TI2MMesh.CutOut), lit, with its edges drawn over it as
  a wireframe pass of the same triangles; opacity is a uniform, so the slider
  needs no rebuild.

  One coordinate frame: the image's voxels scaled by its voxel size, i.e.
  millimetres from the image's corner, [0, n * d] per axis; a mesh is mapped
  into it by the caller. With no image, the mesh's own coordinates. }
unit i2mview;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, Controls, ExtCtrls, Forms, Graphics, FPImage, FPWritePNG,
  OpenGLContext, GL, GLext, IntfGraphics, GraphType, mcxgl, i2mmesh, i2megl, i2mshapes;

type
  TI2MLog = procedure(Sender: TObject; const AText: string) of object;

  { A triangle / line buffer filled in one go (mcxgl's grow one vertex at a
    time, which is fine for a few shapes and not for a million triangles).
    Same vertex layouts: triangles pos, normal, rgba; lines pos, rgb. }
  TI2MBuffer = class
  private
    FData: array of Single;
    FCount, FStride, FUsed: Integer;
    FLines: Boolean;
    FVAO, FVBO: GLuint;
    FDirty: Boolean;
  public
    constructor Create(ALines: Boolean);
    destructor Destroy; override;
    procedure Clear;
    procedure Reserve(AVertices: Integer);
    procedure Tri(const A, B, C: TI2MPoint; cr, cg, cb, ca: Single);
    procedure Line(const A, B: TI2MPoint; cr, cg, cb: Single);
    procedure Draw; overload;
    procedure Draw(AFirst, ACount: Integer); overload;
    property Count: Integer read FCount;
  end;

  TI2MView = class
  private
    FHost: TWinControl;
    FGL: TOpenGLControl;      { a window with its own GL context (GLX / WGL / ...) }
    FBox: TPaintBox;          { or: an offscreen EGL context, copied in }
    FSurface: TControl;       { whichever of the two }
    FEgl: Boolean;
    FOffscreen: TMcxTarget;
    FFrameBmp: TBitmap;
    FBackend: string;
    FCamera: TMcxCamera;
    FLineShader, FSolidShader, FVolShader: TMcxShader;
    FFrame, FSurf: TI2MBuffer;
    { the shape constructs' preview (i2mshapes): translucent triangles, and
      their domain (the frame when nothing else is shown) }
    FShapeSurf: TI2MBuffer;
    FShapeLo, FShapeHi: TMcxVec3;
    FHasShapes, FShowShapes: Boolean;
    { their triangles, re-sorted far to near whenever the eye moves (translucent
      surfaces that cross blend right only drawn back to front) }
    FShapeTris: array of record
      A, B, C: TI2MPoint;
      cr, cg, cb, ca: Single;
    end;
    FShapeEye: TMcxVec3;
    FShapeSorted: Boolean;
    { a Refresh asked for while painting: queued until the paint is done (GTK
      refuses an invalidation during a paint) }
    FPainting, FRefreshQueued: Boolean;
    FAxisText: TI2MBuffer;      { the axis letters and tick numbers, facing the camera }
    FAxisStep: array[0..2] of Single;
    FOrient: string;            { the anatomical letter at each axis' high end, or '' }
    { the scene (the image's voxel axes, mm) turned to RAS for the display: scene
      axis k is display axis FAx[k], the way FSg[k] (by FOrient; else as it is) }
    FAx, FSg: array[0..2] of Integer;
    FVolume: TMcxVolume;
    FCube: TMcxCube;
    FVolDims: array[0..2] of Integer;
    FVoxel: TMcxVec3;           { the image's voxel size }
    FMesh: TI2MMesh;
    FBoxLo, FBoxHi: TMcxVec3;   { the frame: the image, else the mesh }
    FClipLo, FClipHi: TMcxVec3; { fractions 0..1 of the frame }
    FShowVolume, FShowMesh, FShowEdges: Boolean;
    FMeshAlpha, FOpacity, FFloor: Single;
    FMap, FStyle: Integer;
    FVolLow, FVolHigh: Single;
    FHidden: array of Boolean;  { per label }
    FBack: TMcxVec3;
    FReady, FFailed, FDragging, FCoarse: Boolean;
    FDragX, FDragY: Integer;
    FPanX, FPanY: Single;       { the pan: a shift of the view (mm, screen axes); the camera
                                  keeps orbiting the frame's centre }
    FPanning: Boolean;
    FOnLog: TI2MLog;
    FDescription: string;
    procedure AsyncRefresh(Data: PtrInt);
    procedure SortShapes(const AEye: TMcxVec3);
    procedure BoxPaintFrame;
    procedure Say(const AText: string);
    function Current: Boolean;
    procedure Refresh;
    procedure BoxPaint(Sender: TObject);
    function Start: Boolean;
    procedure GLPaint(Sender: TObject);
    procedure GLMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure GLMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure GLMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
    procedure GLMouseWheel(Sender: TObject; Shift: TShiftState; WheelDelta: Integer;
      MousePos: TPoint; var Handled: Boolean);
    procedure BuildFrame;
    procedure BuildAxisLabels;
    procedure SetOrientation(const AValue: string);
    procedure BuildSurface;
    procedure UpdateFrameBox;
    function ClipBox(out ALo, AHi: TMcxVec3): Boolean;
    function ViewEye: TMcxVec3;
    function ToDisplay(const V: TMcxVec3): TMcxVec3;
    function ToScene(const V: TMcxVec3): TMcxVec3;
    function Reoriented: Boolean;
    procedure RenderScene(AWidth, AHeight: Integer);
    procedure DrawVolume(const AMVP: TMcxMat4);
    function GetLabelVisible(ATag: Integer): Boolean;
    procedure SetLabelVisible(ATag: Integer; AValue: Boolean);
  public
    { AMode: 'auto' (a GL window if the display has a GL visual, else EGL),
      'glx' (always a GL window), 'egl' (always offscreen EGL: the GPU, else
      software), 'soft' (offscreen, Mesa's software rasteriser) }
    constructor Create(AHost: TWinControl; const AMode: string = 'auto');
    destructor Destroy; override;
    { AData x fastest, one channel; AVoxel the voxel size (mm). }
    function SetVolume(AData: PSingle; ANx, ANy, ANz: Integer; ALow, AHigh: Single;
      const AVoxel: TMcxVec3): Boolean;
    procedure ClearVolume;
    { The view does not own the mesh. }
    procedure SetMesh(AMesh: TI2MMesh);
    { the shape constructs' preview (drawn while ShowShapes) }
    procedure SetShapes(const AScene: TI2MShapeScene);
    procedure ClearShapes;
    procedure SetShowShapes(AValue: Boolean);
    property ShowShapes: Boolean read FShowShapes write SetShowShapes;
    procedure MeshChanged;   { labels shown: the cut-out again }
    procedure ClipChanged;   { the box moved: the cut-out again }
    { frames the box; ALeft / ARight: pixels at either side panels cover, left out }
    procedure FitView(ALeft: Integer = 0; ARight: Integer = 0);
    { the camera's default angles: from the front -- the image's anterior side
      (by its orientation letters), else +y (RAS) -- a little to the right and
      from above }
    procedure DefaultAngles;
    procedure Redraw;
    function SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
    function HasVolume: Boolean;
    { an image or a mesh to show (else the view is left empty: no frame) }
    function HasContent: Boolean;
    { the view's background, for what is laid over it }
    function BackgroundColor: TColor;
    property ShowVolume: Boolean read FShowVolume write FShowVolume;
    property ShowMesh: Boolean read FShowMesh write FShowMesh;
    property ShowEdges: Boolean read FShowEdges write FShowEdges;
    property MeshAlpha: Single read FMeshAlpha write FMeshAlpha;
    property Opacity: Single read FOpacity write FOpacity;
    property Threshold: Single read FFloor write FFloor;
    property Colormap: Integer read FMap write FMap;
    property Style: Integer read FStyle write FStyle;
    property ClipLo: TMcxVec3 read FClipLo write FClipLo;
    property ClipHi: TMcxVec3 read FClipHi write FClipHi;
    property LabelVisible[ATag: Integer]: Boolean read GetLabelVisible write SetLabelVisible;
    property Background: TMcxVec3 read FBack write FBack;
    { 'RAS'-style: the letter at the high end of x, y and z ('' = none) }
    property Orientation: string read FOrient write SetOrientation;
    property Description: string read FDescription;
    property Backend: string read FBackend;
    property OnLog: TI2MLog read FOnLog write FOnLog;
  end;

{ The colour of a label: a fixed palette (label 0 grey). }
procedure I2MLabelColour(ATag: Integer; out r, g, b: Single);

implementation

const
  AxisCol: array[0..2, 0..2] of Single = ((0.90, 0.30, 0.25), (0.35, 0.75, 0.35), (0.35, 0.55, 0.95));
  Palette: array[0..11, 0..2] of Single = (
    (0.62, 0.62, 0.62), (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76),
    (0.74, 0.74, 0.13), (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.00, 0.73, 0.47));

  { the mesh: mcxgl's lit two-sided solid, with the opacity a uniform, and a
    wireframe pass (uWire) in a darker shade of each face's colour }
  MeshVS =
    '#version 330 core'#10 +
    'layout(location = 0) in vec3 aPos;'#10 +
    'layout(location = 1) in vec3 aNormal;'#10 +
    'layout(location = 2) in vec4 aColour;'#10 +
    'uniform mat4 uMVP;'#10 +
    'out vec3 vNormal; out vec4 vColour;'#10 +
    'void main() { vNormal = aNormal; vColour = aColour;'#10 +
    '  gl_Position = uMVP * vec4(aPos, 1.0); }'#10;
  MeshFS =
    '#version 330 core'#10 +
    'in vec3 vNormal; in vec4 vColour;'#10 +
    'out vec4 oColour;'#10 +
    'uniform vec3 uLight; uniform float uAlpha; uniform int uWire; uniform float uWireAlpha;'#10 +
    'void main() {'#10 +
    '  if (uWire == 1) { oColour = vec4(vColour.rgb * 0.25, uWireAlpha); return; }'#10 +
    '  float d = abs(dot(normalize(vNormal), normalize(uLight)));'#10 +
    '  oColour = vec4(vColour.rgb * (0.38 + 0.62 * d), uAlpha * vColour.a); }'#10;   { (a vertex''s own alpha: the shapes'') }
  ClipLineVS =
    '#version 330 core'#10 +
    'layout(location = 0) in vec3 aPos;'#10 +
    'layout(location = 1) in vec3 aColour;'#10 +
    'uniform mat4 uMVP;'#10 +
    'out vec3 vPos; out vec3 vColour;'#10 +
    'void main() { vPos = aPos; vColour = aColour; gl_Position = uMVP * vec4(aPos, 1.0); }'#10;
  ClipLineFS =
    '#version 330 core'#10 +
    'in vec3 vPos; in vec3 vColour;'#10 +
    'out vec4 oColour;'#10 +
    'uniform vec3 uLo; uniform vec3 uHi; uniform int uSkip; uniform float uAlpha; uniform float uEps;'#10 +
    'void main() {'#10 +
    '  for (int a = 0; a < 3; a++)'#10 +
    '    if (a != uSkip && (vPos[a] < uLo[a] - uEps || vPos[a] > uHi[a] + uEps)) discard;'#10 +
    '  oColour = vec4(vColour, uAlpha); }'#10;

procedure I2MLabelColour(ATag: Integer; out r, g, b: Single);
var
  k: Integer;
begin
  k := Abs(ATag) mod Length(Palette);
  r := Palette[k, 0];
  g := Palette[k, 1];
  b := Palette[k, 2];
end;

function P3(x, y, z: Single): TI2MPoint;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

{ ------------------------------------------------------------ TI2MBuffer --- }

constructor TI2MBuffer.Create(ALines: Boolean);
begin
  FLines := ALines;
  if ALines then FStride := 6 else FStride := 10;
end;

destructor TI2MBuffer.Destroy;
begin
  if FVBO <> 0 then glDeleteBuffers(1, @FVBO);
  if FVAO <> 0 then glDeleteVertexArrays(1, @FVAO);
  inherited Destroy;
end;

procedure TI2MBuffer.Clear;
begin
  FCount := 0;
  FUsed := 0;
  FDirty := True;
end;

procedure TI2MBuffer.Reserve(AVertices: Integer);
begin
  if (FUsed + AVertices) * FStride > Length(FData) then
    SetLength(FData, Max((FUsed + AVertices) * FStride, 2 * Length(FData)));
end;

procedure TI2MBuffer.Tri(const A, B, C: TI2MPoint; cr, cg, cb, ca: Single);
var
  ux, uy, uz, vx, vy, vz, nx, ny, nz, l: Single;
  k, o: Integer;
  P: array[0..2] of TI2MPoint;
begin
  Reserve(3);
  ux := B.x - A.x; uy := B.y - A.y; uz := B.z - A.z;
  vx := C.x - A.x; vy := C.y - A.y; vz := C.z - A.z;
  nx := uy * vz - uz * vy;
  ny := uz * vx - ux * vz;
  nz := ux * vy - uy * vx;
  l := Sqrt(nx * nx + ny * ny + nz * nz);
  if l > 0 then
  begin
    nx := nx / l; ny := ny / l; nz := nz / l;
  end;
  P[0] := A; P[1] := B; P[2] := C;
  for k := 0 to 2 do
  begin
    o := FUsed * FStride;
    FData[o] := P[k].x; FData[o + 1] := P[k].y; FData[o + 2] := P[k].z;
    FData[o + 3] := nx; FData[o + 4] := ny; FData[o + 5] := nz;
    FData[o + 6] := cr; FData[o + 7] := cg; FData[o + 8] := cb; FData[o + 9] := ca;
    Inc(FUsed);
  end;
  FCount := FUsed;
  FDirty := True;
end;

procedure TI2MBuffer.Line(const A, B: TI2MPoint; cr, cg, cb: Single);
var
  o: Integer;
begin
  Reserve(2);
  o := FUsed * FStride;
  FData[o] := A.x; FData[o + 1] := A.y; FData[o + 2] := A.z;
  FData[o + 3] := cr; FData[o + 4] := cg; FData[o + 5] := cb;
  FData[o + 6] := B.x; FData[o + 7] := B.y; FData[o + 8] := B.z;
  FData[o + 9] := cr; FData[o + 10] := cg; FData[o + 11] := cb;
  Inc(FUsed, 2);
  FCount := FUsed;
  FDirty := True;
end;

procedure TI2MBuffer.Draw;
begin
  Draw(0, FCount);
end;

procedure TI2MBuffer.Draw(AFirst, ACount: Integer);
begin
  if (FCount = 0) or (ACount <= 0) then Exit;
  if FVAO = 0 then
  begin
    glGenVertexArrays(1, @FVAO);
    glGenBuffers(1, @FVBO);
  end;
  glBindVertexArray(FVAO);
  if FDirty then
  begin
    glBindBuffer(GL_ARRAY_BUFFER, FVBO);
    glBufferData(GL_ARRAY_BUFFER, FUsed * FStride * SizeOf(Single), @FData[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, FStride * SizeOf(Single), nil);
    glEnableVertexAttribArray(0);
    if FLines then
    begin
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, FStride * SizeOf(Single), Pointer(3 * SizeOf(Single)));
      glEnableVertexAttribArray(1);
    end
    else
    begin
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, FStride * SizeOf(Single), Pointer(3 * SizeOf(Single)));
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, FStride * SizeOf(Single), Pointer(6 * SizeOf(Single)));
      glEnableVertexAttribArray(2);
    end;
    FDirty := False;
  end;
  if FLines then glDrawArrays(GL_LINES, AFirst, ACount)
  else glDrawArrays(GL_TRIANGLES, AFirst, ACount);
  glBindVertexArray(0);
end;

{ -------------------------------------------------------------- TI2MView --- }

constructor TI2MView.Create(AHost: TWinControl; const AMode: string);
var
  Desc: string;
  Samples, k: Integer;
begin
  FHost := AHost;
  FCamera := TMcxCamera.Create;
  FFrame := TI2MBuffer.Create(True);
  FAxisText := TI2MBuffer.Create(True);
  FSurf := TI2MBuffer.Create(False);
  FShapeSurf := TI2MBuffer.Create(False);
  FShowVolume := True;
  FShowMesh := True;
  FShowEdges := True;
  FMeshAlpha := 1;
  FVoxel := McxVec3(1, 1, 1);
  FOpacity := 0.5;
  FFloor := 0.02;
  FMap := 2;
  FStyle := 1;
  FClipLo := McxVec3(0, 0, 0);
  FClipHi := McxVec3(1, 1, 1);
  FBoxLo := McxVec3(0, 0, 0);
  FBoxHi := McxVec3(1, 1, 1);
  for k := 0 to 2 do
  begin
    FAx[k] := k;
    FSg[k] := 1;
  end;
  FPanX := 0;
  FPanY := 0;
  FBack := McxVec3(0.08, 0.09, 0.11);

  Samples := 4;
  FEgl := (AMode = 'egl') or (AMode = 'soft');
  if AMode = 'auto' then
  begin
    { the LCL's GL control raises inside handle creation when the display has
      no GL visual (X2Go, NX, VNC, ...): look first }
    if not I2MGlxUsable(Samples) then
    begin
      Samples := 1;
      FEgl := not I2MGlxUsable(1);
    end;
  end;
  if FEgl then
  begin
    if AMode = 'soft' then FEgl := I2MEglOpen(emSoftware, Desc)
    else FEgl := I2MEglOpen(emAuto, Desc);
    if FEgl then
      FBackend := 'offscreen, ' + Desc
    else
      FBackend := 'no GL window and no EGL (' + Desc + ')';
  end;
  if FEgl then
  begin
    FBox := TPaintBox.Create(FHost);
    FBox.Parent := FHost;
    FBox.Align := alClient;
    FBox.OnPaint := @BoxPaint;
    FBox.OnMouseDown := @GLMouseDown;
    FBox.OnMouseMove := @GLMouseMove;
    FBox.OnMouseUp := @GLMouseUp;
    FBox.OnMouseWheel := @GLMouseWheel;
    FSurface := FBox;
    FFrameBmp := TBitmap.Create;
    Exit;
  end;
  if FBackend = '' then FBackend := 'GL window';
  FGL := TOpenGLControl.Create(FHost);
  FGL.Parent := FHost;
  FGL.Align := alClient;
  FGL.DepthBits := 24;
  FGL.MultiSampling := Samples;
  FGL.OpenGLMajorVersion := 3;
  FGL.OpenGLMinorVersion := 3;
  FGL.OnPaint := @GLPaint;
  FGL.OnMouseDown := @GLMouseDown;
  FGL.OnMouseMove := @GLMouseMove;
  FGL.OnMouseUp := @GLMouseUp;
  FGL.OnMouseWheel := @GLMouseWheel;
  FSurface := FGL;
end;

destructor TI2MView.Destroy;
begin
  if Current then
  begin
    FreeAndNil(FVolume);
    FreeAndNil(FCube);
    FreeAndNil(FFrame);
    FreeAndNil(FAxisText);
    FreeAndNil(FSurf);
    FreeAndNil(FShapeSurf);
    Application.RemoveAsyncCalls(Self);
    FreeAndNil(FLineShader);
    FreeAndNil(FSolidShader);
    FreeAndNil(FVolShader);
    FreeAndNil(FOffscreen);
  end;
  if FEgl then I2MEglClose;
  FFrameBmp.Free;
  FCamera.Free;
  inherited Destroy;
end;

function TI2MView.Current: Boolean;
begin
  if FEgl then Result := I2MEglMakeCurrent
  else Result := (FGL <> nil) and FGL.MakeCurrent;
end;

procedure TI2MView.Refresh;
begin
  if FPainting then
  begin   { (GTK refuses an invalidation during a paint: after it) }
    if not FRefreshQueued then
    begin
      FRefreshQueued := True;
      Application.QueueAsyncCall(@AsyncRefresh, 0);
    end;
    Exit;
  end;
  if FSurface <> nil then FSurface.Invalidate;
end;

procedure TI2MView.AsyncRefresh(Data: PtrInt);
begin
  FRefreshQueued := False;
  Refresh;
end;

{ offscreen: render into a framebuffer object, read it back, show it }
procedure TI2MView.BoxPaint(Sender: TObject);
begin
  FPainting := True;
  try
    BoxPaintFrame;
  finally
    FPainting := False;
  end;
end;

procedure TI2MView.BoxPaintFrame;
var
  W, H, y: Integer;
  Img: TLazIntfImage;
  Px: array of Byte;
  Desc: TRawImageDescription;
begin
  W := FBox.Width;
  H := FBox.Height;
  if (W < 1) or (H < 1) then Exit;
  if not Current or not Start then
  begin
    FBox.Canvas.Brush.Color := RGBToColor(Round(FBack.x * 255), Round(FBack.y * 255), Round(FBack.z * 255));
    FBox.Canvas.FillRect(0, 0, W, H);
    FBox.Canvas.Font.Color := clWhite;
    FBox.Canvas.TextOut(10, 10, 'no OpenGL: ' + FBackend);
    Exit;
  end;
  if FOffscreen = nil then FOffscreen := TMcxTarget.Create;
  if not FOffscreen.Bind(W, H) then
  begin
    FOffscreen.Unbind;
    Exit;
  end;
  RenderScene(W, H);
  SetLength(Px, W * H * 4);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, W, H, GL_BGRA, GL_UNSIGNED_BYTE, @Px[0]);
  FOffscreen.Unbind;
  Img := TLazIntfImage.Create(0, 0);
  try
    { (a local: DataDescription is a record property, and a method called
      on it would change a temporary) }
    Desc.Init_BPP32_B8G8R8_BIO_TTB(W, H);   { opaque: the FBO's alpha is blending's, not the picture's }
    Img.DataDescription := Desc;
    for y := 0 to H - 1 do   { GL's rows run bottom up }
      Move(Px[(H - 1 - y) * W * 4], Img.GetDataLineStart(y)^, W * 4);
    FFrameBmp.LoadFromIntfImage(Img);
  finally
    Img.Free;
  end;
  FBox.Canvas.Draw(0, 0, FFrameBmp);
  if FCoarse then
  begin
    FCoarse := False;
    Refresh;
  end;
end;

procedure TI2MView.Say(const AText: string);
begin
  if Assigned(FOnLog) then FOnLog(Self, AText);
end;

function TI2MView.Start: Boolean;
begin
  Result := FReady;
  if FReady or FFailed then Exit;
  if not McxGLLoad then
  begin
    FFailed := True;
    Say('OpenGL 3.3 is not available: ' + McxGLDescribe);
    Exit(False);
  end;
  FLineShader := TMcxShader.Create;
  FSolidShader := TMcxShader.Create;
  FVolShader := TMcxShader.Create;
  if not FLineShader.Build(ClipLineVS, ClipLineFS) then
  begin
    Say('line shader: ' + FLineShader.Error);
    FFailed := True;
    Exit(False);
  end;
  if not FSolidShader.Build(MeshVS, MeshFS) then
  begin
    Say('surface shader: ' + FSolidShader.Error);
    FFailed := True;
    Exit(False);
  end;
  if not FVolShader.Build(McxVolumeVertexShader, McxVolumeFragmentShader) then
  begin
    Say('volume shader: ' + FVolShader.Error);
    FreeAndNil(FVolShader);
  end;
  FDescription := McxGLDescribe;
  Say('OpenGL: ' + FDescription);
  FReady := True;
  Result := True;
end;

function TI2MView.HasVolume: Boolean;
begin
  Result := (FVolume <> nil) and FVolume.Loaded;
end;

function TI2MView.HasContent: Boolean;
begin
  Result := HasVolume or ((FMesh <> nil) and (FMesh.NodeCount > 0)) or (FHasShapes and FShowShapes);
end;

procedure TI2MView.SetShapes(const AScene: TI2MShapeScene);
var
  i: Integer;
  r, g, b: Single;
begin
  SetLength(FShapeTris, Length(AScene.Tris));
  for i := 0 to High(AScene.Tris) do
  begin
    I2MLabelColour(AScene.Tris[i].Tag, r, g, b);
    FShapeTris[i].A := AScene.Tris[i].A;
    FShapeTris[i].B := AScene.Tris[i].B;
    FShapeTris[i].C := AScene.Tris[i].C;
    FShapeTris[i].cr := r;
    FShapeTris[i].cg := g;
    FShapeTris[i].cb := b;
    FShapeTris[i].ca := AScene.Tris[i].Alpha;
  end;
  FShapeSorted := False;
  FShapeLo := McxVec3(AScene.Lo.x, AScene.Lo.y, AScene.Lo.z);
  FShapeHi := McxVec3(AScene.Hi.x, AScene.Hi.y, AScene.Hi.z);
  FHasShapes := True;
  UpdateFrameBox;
  Refresh;
end;

procedure TI2MView.ClearShapes;
begin
  FShapeSurf.Clear;
  FShapeTris := nil;
  FShapeSorted := False;
  FHasShapes := False;
  UpdateFrameBox;
  Refresh;
end;

{ the shape triangles far to near from AEye (by their centroids), into the buffer }
procedure TI2MView.SortShapes(const AEye: TMcxVec3);
var
  Key: array of Single;
  Idx: array of Integer;
  i: Integer;

  procedure QSort(L, R: Integer);
  var
    i, j, t: Integer;
    p: Single;
  begin
    while L < R do
    begin
      i := L;
      j := R;
      p := Key[Idx[(L + R) shr 1]];
      repeat
        while Key[Idx[i]] > p do Inc(i);
        while Key[Idx[j]] < p do Dec(j);
        if i <= j then
        begin
          t := Idx[i];
          Idx[i] := Idx[j];
          Idx[j] := t;
          Inc(i);
          Dec(j);
        end;
      until i > j;
      if j - L < R - i then
      begin
        QSort(L, j);
        L := i;
      end
      else
      begin
        QSort(i, R);
        R := j;
      end;
    end;
  end;

begin
  SetLength(Key, Length(FShapeTris));
  SetLength(Idx, Length(FShapeTris));
  for i := 0 to High(FShapeTris) do
    with FShapeTris[i] do
    begin
      Key[i] := Sqr((A.x + B.x + C.x) / 3 - AEye.x) + Sqr((A.y + B.y + C.y) / 3 - AEye.y) +
        Sqr((A.z + B.z + C.z) / 3 - AEye.z);
      Idx[i] := i;
    end;
  if Length(Idx) > 1 then QSort(0, High(Idx));   { (descending: the farthest first) }
  FShapeSurf.Clear;
  FShapeSurf.Reserve(3 * Length(FShapeTris));
  for i := 0 to High(Idx) do
    with FShapeTris[Idx[i]] do FShapeSurf.Tri(A, B, C, cr, cg, cb, ca);
  FShapeEye := AEye;
  FShapeSorted := True;
end;

procedure TI2MView.SetShowShapes(AValue: Boolean);
begin
  if FShowShapes = AValue then Exit;
  FShowShapes := AValue;
  UpdateFrameBox;
  Refresh;
end;

function TI2MView.BackgroundColor: TColor;
begin
  Result := RGBToColor(Round(FBack.x * 255), Round(FBack.y * 255), Round(FBack.z * 255));
end;

procedure TI2MView.UpdateFrameBox;
begin
  if HasVolume then
  begin
    FBoxLo := McxVec3(0, 0, 0);
    FBoxHi := McxVec3(FVolDims[0] * FVoxel.x, FVolDims[1] * FVoxel.y, FVolDims[2] * FVoxel.z);
  end
  else if (FMesh <> nil) and (FMesh.NodeCount > 0) then
  begin
    FBoxLo := McxVec3(FMesh.Lo.x, FMesh.Lo.y, FMesh.Lo.z);
    FBoxHi := McxVec3(FMesh.Hi.x, FMesh.Hi.y, FMesh.Hi.z);
  end
  else if FHasShapes and FShowShapes then
  begin
    FBoxLo := FShapeLo;
    FBoxHi := FShapeHi;
  end;
  { the camera orbits the frame's centre }
  FCamera.Target := ToDisplay(McxVec3((FBoxLo.x + FBoxHi.x) / 2, (FBoxLo.y + FBoxHi.y) / 2, (FBoxLo.z + FBoxHi.z) / 2));
  BuildFrame;
end;

function TI2MView.SetVolume(AData: PSingle; ANx, ANy, ANz: Integer; ALow, AHigh: Single;
  const AVoxel: TMcxVec3): Boolean;
begin
  Result := False;
  if not Current then Exit;
  if not Start then Exit;
  if AHigh <= ALow then AHigh := ALow + 1;
  FVolLow := ALow;
  FVolHigh := AHigh;
  if FVolume = nil then FVolume := TMcxVolume.Create;
  Result := FVolume.Upload(AData, ANx, ANy, ANz, ALow, AHigh);
  if not Result then
  begin
    Say(Format('the GPU would not take a %dx%dx%d volume', [ANx, ANy, ANz]));
    FreeAndNil(FVolume);
    Exit;
  end;
  FVolDims[0] := ANx;
  FVolDims[1] := ANy;
  FVolDims[2] := ANz;
  FVoxel := AVoxel;
  if (FVoxel.x <= 0) or (FVoxel.y <= 0) or (FVoxel.z <= 0) then FVoxel := McxVec3(1, 1, 1);
  UpdateFrameBox;
  Refresh;
end;

procedure TI2MView.ClearVolume;
begin
  if Current then FreeAndNil(FVolume);
  UpdateFrameBox;
  Refresh;
end;

procedure TI2MView.SetMesh(AMesh: TI2MMesh);
begin
  FMesh := AMesh;
  if not Current then Exit;
  if not Start then Exit;
  UpdateFrameBox;
  BuildSurface;
  Refresh;
end;

procedure TI2MView.MeshChanged;
begin
  if not Current or not FReady then Exit;
  BuildSurface;
  Refresh;
end;

procedure TI2MView.ClipChanged;
begin
  if not Current or not FReady then Exit;
  BuildFrame;
  BuildSurface;
  Refresh;
end;

function TI2MView.GetLabelVisible(ATag: Integer): Boolean;
begin
  Result := (ATag < 0) or (ATag > High(FHidden)) or not FHidden[ATag];
end;

procedure TI2MView.SetLabelVisible(ATag: Integer; AValue: Boolean);
begin
  if ATag < 0 then Exit;
  if ATag > High(FHidden) then SetLength(FHidden, ATag + 1);
  FHidden[ATag] := not AValue;
end;

{ the frame: the image (or mesh) box, and the crop box inside it }
{ a round step for a ruler over ASpan: 1, 2 or 5 times a power of ten, the
  largest giving 5 to 12 divisions (mcxstudio2's McxNiceStep allows 10.5, and
  a 219 mm image, 10.95 steps of 20 or 4.4 of 50, fell through to span / 8:
  ticks at 27.38, 54.75, ..) }
function NiceStep(ASpan: Single): Single;
const
  Nice: array[0..2] of Single = (1, 2, 5);
var
  k, i: Integer;
  St, n: Double;
begin
  Result := 1;
  if ASpan <= 0 then Exit;
  Result := 0;
  for k := -4 to 8 do
    for i := 0 to 2 do
    begin
      St := Nice[i] * Power(10, k);
      n := ASpan / St;
      if (n >= 5) and (n <= 12) and (St > Result) then Result := St;
    end;
  if Result = 0 then Result := ASpan / 8;
end;

procedure TI2MView.BuildFrame;
var
  L, H: TMcxVec3;
  A0, B0: TI2MPoint;
  a, k: Integer;
  lo, hi, st, t, Tick: Single;

  procedure Box(const A, B: TMcxVec3; r, g, bl: Single);
  var
    c: array[0..7] of TI2MPoint;
    i: Integer;
  begin
    for i := 0 to 7 do
      c[i] := P3(IfThen(i and 1 <> 0, B.x, A.x), IfThen(i and 2 <> 0, B.y, A.y), IfThen(i and 4 <> 0, B.z, A.z));
    FFrame.Line(c[0], c[1], r, g, bl); FFrame.Line(c[2], c[3], r, g, bl);
    FFrame.Line(c[4], c[5], r, g, bl); FFrame.Line(c[6], c[7], r, g, bl);
    FFrame.Line(c[0], c[2], r, g, bl); FFrame.Line(c[1], c[3], r, g, bl);
    FFrame.Line(c[4], c[6], r, g, bl); FFrame.Line(c[5], c[7], r, g, bl);
    FFrame.Line(c[0], c[4], r, g, bl); FFrame.Line(c[1], c[5], r, g, bl);
    FFrame.Line(c[2], c[6], r, g, bl); FFrame.Line(c[3], c[7], r, g, bl);
  end;

begin
  FFrame.Clear;
  Box(FBoxLo, FBoxHi, 0.85, 0.80, 0.30);
  if ClipBox(L, H) and ((FClipLo.x > 0) or (FClipLo.y > 0) or (FClipLo.z > 0) or
     (FClipHi.x < 1) or (FClipHi.y < 1) or (FClipHi.z < 1)) then
    Box(L, H, 0.45, 0.75, 0.95);
  { the axes, as mcxstudio2's: x red, y green, z blue, the whole length of
    the frame from its low corner, a tick at every round step (x's toward -y,
    y's and z's toward -x: where their numbers go) }
  Tick := Max(FBoxHi.x - FBoxLo.x, Max(FBoxHi.y - FBoxLo.y, FBoxHi.z - FBoxLo.z)) * 0.025;
  for a := 0 to 2 do
  begin
    case a of
      0: begin lo := FBoxLo.x; hi := FBoxHi.x; end;
      1: begin lo := FBoxLo.y; hi := FBoxHi.y; end;
    else begin lo := FBoxLo.z; hi := FBoxHi.z; end;
    end;
    st := NiceStep(hi - lo);
    FAxisStep[a] := st;
    A0 := P3(FBoxLo.x, FBoxLo.y, FBoxLo.z);
    B0 := A0;
    case a of
      0: B0.x := hi;
      1: B0.y := hi;
    else B0.z := hi;
    end;
    FFrame.Line(A0, B0, AxisCol[a, 0], AxisCol[a, 1], AxisCol[a, 2]);
    k := Ceil((lo - st * 0.001) / st);
    while k * st <= hi + st * 0.001 do
    begin
      t := k * st;
      A0 := P3(FBoxLo.x, FBoxLo.y, FBoxLo.z);
      B0 := A0;
      case a of
        0: begin A0.x := t; B0.x := t; B0.y := FBoxLo.y - Tick; end;
        1: begin A0.y := t; B0.y := t; B0.x := FBoxLo.x - Tick; end;
      else begin A0.z := t; B0.z := t; B0.x := FBoxLo.x - Tick; end;
      end;
      FFrame.Line(A0, B0, AxisCol[a, 0], AxisCol[a, 1], AxisCol[a, 2]);
      Inc(k);
    end;
  end;
end;

{ the letters (RAS-style, at each axis' high end): the display turned so R, A
  and S are +x, +y and +z -- the head upright, its face to the front; letters
  that are not one of R/L, A/P and S/I each: the scene as it is }
procedure TI2MView.SetOrientation(const AValue: string);
var
  k, a, sg: Integer;
  Used: array[0..2] of Boolean;
  Ok: Boolean;
  c: Char;
begin
  FOrient := AValue;
  Ok := Length(AValue) = 3;
  for k := 0 to 2 do Used[k] := False;
  for k := 0 to 2 do
  begin
    FAx[k] := k;
    FSg[k] := 1;
  end;
  if Ok then
    for k := 0 to 2 do
    begin
      c := UpCase(AValue[k + 1]);
      a := Pos(c, 'RAS') - 1;
      sg := 1;
      if a < 0 then
      begin
        a := Pos(c, 'LPI') - 1;
        sg := -1;
      end;
      if (a < 0) or Used[a] then
      begin
        Ok := False;
        Break;
      end;
      Used[a] := True;
      FAx[k] := a;
      FSg[k] := sg;
    end;
  if not Ok then
    for k := 0 to 2 do
    begin
      FAx[k] := k;
      FSg[k] := 1;
    end;
  if FCamera <> nil then   { (the camera orbits the frame's centre, on the display) }
    FCamera.Target := ToDisplay(McxVec3((FBoxLo.x + FBoxHi.x) / 2, (FBoxLo.y + FBoxHi.y) / 2, (FBoxLo.z + FBoxHi.z) / 2));
  Refresh;
end;

function TI2MView.Reoriented: Boolean;
begin
  Result := (FAx[0] <> 0) or (FAx[1] <> 1) or (FSg[0] < 0) or (FSg[1] < 0) or (FSg[2] < 0);
end;

{ a scene point / direction on the display, and back }
function TI2MView.ToDisplay(const V: TMcxVec3): TMcxVec3;
var
  P: array[0..2] of Single;
  D: array[0..2] of Single;
  k: Integer;
begin
  P[0] := V.x; P[1] := V.y; P[2] := V.z;
  for k := 0 to 2 do D[FAx[k]] := FSg[k] * P[k];
  Result := McxVec3(D[0], D[1], D[2]);
end;

function TI2MView.ToScene(const V: TMcxVec3): TMcxVec3;
var
  P: array[0..2] of Single;
  D: array[0..2] of Single;
  k: Integer;
begin
  D[0] := V.x; D[1] := V.y; D[2] := V.z;
  for k := 0 to 2 do P[k] := FSg[k] * D[FAx[k]];
  Result := McxVec3(P[0], P[1], P[2]);
end;

{ The letters and numbers, turned to face the camera (rebuilt every frame):
  stroked, as mcxstudio2's -- a core profile has no text, and a dozen glyphs
  do not justify a font atlas. Digits are seven-segment. }
procedure TI2MView.BuildAxisLabels;
const
  { u, v pairs in a unit box, two per stroke }
  GX: array[0..7] of Single = (0, 0, 1, 1,  0, 1, 1, 0);
  GY: array[0..11] of Single = (0, 1, 0.5, 0.5,  1, 1, 0.5, 0.5,  0.5, 0.5, 0.5, 0);
  GZ: array[0..11] of Single = (0, 1, 1, 1,  1, 1, 0, 0,  0, 0, 1, 0);
  GR: array[0..27] of Single = (0, 0, 0, 1,  0, 1, 0.75, 1,  0.75, 1, 0.9, 0.85,  0.9, 0.85, 0.9, 0.65,
    0.9, 0.65, 0.75, 0.5,  0.75, 0.5, 0, 0.5,  0.35, 0.5, 0.9, 0);
  GA: array[0..11] of Single = (0, 0, 0.45, 1,  0.45, 1, 0.9, 0,  0.18, 0.4, 0.72, 0.4);
  GS: array[0..19] of Single = (0.9, 1, 0, 1,  0, 1, 0, 0.5,  0, 0.5, 0.9, 0.5,  0.9, 0.5, 0.9, 0,  0.9, 0, 0, 0);
  GL: array[0..7] of Single = (0, 1, 0, 0,  0, 0, 0.85, 0);
  GP: array[0..23] of Single = (0, 0, 0, 1,  0, 1, 0.75, 1,  0.75, 1, 0.9, 0.85,  0.9, 0.85, 0.9, 0.65,
    0.9, 0.65, 0.75, 0.5,  0.75, 0.5, 0, 0.5);
  GI: array[0..11] of Single = (0.45, 0, 0.45, 1,  0.15, 1, 0.75, 1,  0.15, 0, 0.75, 0);
  Seg: array[0..6, 0..3] of Single = (
    (0, 1, 0.55, 1), (0.55, 1, 0.55, 0.5), (0.55, 0.5, 0.55, 0),
    (0, 0, 0.55, 0), (0, 0.5, 0, 0), (0, 1, 0, 0.5), (0, 0.5, 0.55, 0.5));
  Digits: array[0..9, 0..6] of Boolean = (
    (True,  True,  True,  True,  True,  True,  False),
    (False, True,  True,  False, False, False, False),
    (True,  True,  False, True,  True,  False, True),
    (True,  True,  True,  True,  False, False, True),
    (False, True,  True,  False, False, True,  True),
    (True,  False, True,  True,  False, True,  True),
    (True,  False, True,  True,  True,  True,  True),
    (True,  True,  True,  False, False, False, False),
    (True,  True,  True,  True,  True,  True,  True),
    (True,  True,  True,  True,  False, True,  True));
var
  R, U: TMcxVec3;
  Tick, Big, Small: Single;
  a: Integer;
  col: array[0..2] of Single;

  procedure Stroke(const C: TMcxVec3; sc, u0, v0, u1, v1: Single);
  begin
    FAxisText.Line(
      P3(C.x + (R.x * u0 + U.x * v0) * sc, C.y + (R.y * u0 + U.y * v0) * sc, C.z + (R.z * u0 + U.z * v0) * sc),
      P3(C.x + (R.x * u1 + U.x * v1) * sc, C.y + (R.y * u1 + U.y * v1) * sc, C.z + (R.z * u1 + U.z * v1) * sc),
      col[0], col[1], col[2]);
  end;

  procedure Glyph(const ACentre: TMcxVec3; sc: Single; const G: array of Single);
  var
    k: Integer;
    C: TMcxVec3;
  begin
    C := McxVec3(ACentre.x - (R.x + U.x) * sc * 0.5, ACentre.y - (R.y + U.y) * sc * 0.5,
      ACentre.z - (R.z + U.z) * sc * 0.5);
    k := 0;
    while k + 3 <= High(G) do
    begin
      Stroke(C, sc, G[k], G[k + 1], G[k + 2], G[k + 3]);
      Inc(k, 4);
    end;
  end;

  procedure Letter(const ACentre: TMcxVec3; sc: Single; ch: Char);
  begin
    case ch of
      'X': Glyph(ACentre, sc, GX);
      'Y': Glyph(ACentre, sc, GY);
      'Z': Glyph(ACentre, sc, GZ);
      'R': Glyph(ACentre, sc, GR);
      'A': Glyph(ACentre, sc, GA);
      'S': Glyph(ACentre, sc, GS);
      'L': Glyph(ACentre, sc, GL);
      'P': Glyph(ACentre, sc, GP);
      'I': Glyph(ACentre, sc, GI);
    end;
  end;

  procedure Number(const APos: TMcxVec3; AValue: Double; sc: Single);
  var
    S: string;
    i, k, d: Integer;
    C: TMcxVec3;
    Pitch, Left: Single;
  begin
    if Abs(AValue) < 1e-9 then AValue := 0;
    if Abs(AValue - Round(AValue)) < 1e-6 then S := IntToStr(Round(AValue))
    else S := FormatFloat('0.##', AValue);
    Pitch := sc * 0.72;
    Left := -Pitch * (Length(S) - 1) * 0.5 - sc * 0.275;
    for i := 1 to Length(S) do
    begin
      C := McxVec3(APos.x + R.x * (Left + (i - 1) * Pitch) - U.x * sc * 0.5,
        APos.y + R.y * (Left + (i - 1) * Pitch) - U.y * sc * 0.5,
        APos.z + R.z * (Left + (i - 1) * Pitch) - U.z * sc * 0.5);
      if S[i] = '-' then Stroke(C, sc, Seg[6, 0], Seg[6, 1], Seg[6, 2], Seg[6, 3])
      else if S[i] = '.' then Stroke(C, sc, 0.2, 0, 0.35, 0)
      else if S[i] in ['0'..'9'] then
      begin
        d := Ord(S[i]) - Ord('0');
        for k := 0 to 6 do
          if Digits[d, k] then Stroke(C, sc, Seg[k, 0], Seg[k, 1], Seg[k, 2], Seg[k, 3]);
      end;
    end;
  end;

  procedure Numbers(AAxis: Integer);
  var
    lo, hi, st, t: Single;
    P: TMcxVec3;
    k: Integer;
  begin
    case AAxis of
      0: begin lo := FBoxLo.x; hi := FBoxHi.x; end;
      1: begin lo := FBoxLo.y; hi := FBoxHi.y; end;
    else begin lo := FBoxLo.z; hi := FBoxHi.z; end;
    end;
    st := FAxisStep[AAxis];
    if st <= 0 then Exit;
    k := Ceil((lo - st * 0.001) / st);
    while k * st <= hi + st * 0.001 do
    begin
      t := k * st;
      { the far end has the axis letter, and the corner is all three axes' }
      if (t < hi - st * 0.5) and (Abs(t - lo) > st * 0.001) then
      begin
        P := FBoxLo;
        case AAxis of
          0: begin P.x := t; P.y := P.y - Tick * 3; end;
          1: begin P.y := t; P.x := P.x - Tick * 3; end;
        else begin P.z := t; P.x := P.x - Tick * 3; end;
        end;
        Number(P, t, Small);
      end;
      Inc(k);
    end;
  end;

var
  P: TMcxVec3;
begin
  FAxisText.Clear;
  if (FBoxHi.x <= FBoxLo.x) or (FBoxHi.y <= FBoxLo.y) then Exit;
  R := ToScene(FCamera.ScreenRight);   { (the letters face the camera on the display) }
  U := ToScene(FCamera.ScreenUp);
  Tick := Max(FBoxHi.x - FBoxLo.x, Max(FBoxHi.y - FBoxLo.y, FBoxHi.z - FBoxLo.z)) * 0.025;
  Big := Tick * 2.6;
  Small := Tick * 1.7;
  for a := 0 to 2 do
  begin
    col[0] := AxisCol[a, 0];
    col[1] := AxisCol[a, 1];
    col[2] := AxisCol[a, 2];
    P := FBoxLo;
    case a of
      0: P.x := FBoxHi.x + Big;
      1: P.y := FBoxHi.y + Big;
    else P.z := FBoxHi.z + Big;
    end;
    Letter(P, Big, Char(Ord('X') + a));
    { the anatomical direction the axis points to, beside its letter, lighter }
    if Length(FOrient) = 3 then
    begin
      col[0] := 0.5 + 0.5 * AxisCol[a, 0];
      col[1] := 0.5 + 0.5 * AxisCol[a, 1];
      col[2] := 0.5 + 0.5 * AxisCol[a, 2];
      Letter(McxVec3(P.x + R.x * Big * 1.25, P.y + R.y * Big * 1.25, P.z + R.z * Big * 1.25), Big * 0.8, FOrient[a + 1]);
      col[0] := AxisCol[a, 0];
      col[1] := AxisCol[a, 1];
      col[2] := AxisCol[a, 2];
    end;
    Numbers(a);
  end;
end;

{ the eye the pan implies (the ray caster's rays start there): the camera's,
  moved against the pan along the screen's axes }
function TI2MView.ViewEye: TMcxVec3;
var
  E, R, U: TMcxVec3;
begin
  E := FCamera.Eye;
  R := FCamera.ScreenRight;
  U := FCamera.ScreenUp;
  Result := ToScene(McxVec3(E.x - FPanX * R.x - FPanY * U.x, E.y - FPanX * R.y - FPanY * U.y,
    E.z - FPanX * R.z - FPanY * U.z));   { (in the scene: the volume's rays, the shapes' order) }
end;

function TI2MView.ClipBox(out ALo, AHi: TMcxVec3): Boolean;
begin
  ALo := McxVec3(FBoxLo.x + FClipLo.x * (FBoxHi.x - FBoxLo.x), FBoxLo.y + FClipLo.y * (FBoxHi.y - FBoxLo.y),
                 FBoxLo.z + FClipLo.z * (FBoxHi.z - FBoxLo.z));
  AHi := McxVec3(FBoxLo.x + FClipHi.x * (FBoxHi.x - FBoxLo.x), FBoxLo.y + FClipHi.y * (FBoxHi.y - FBoxLo.y),
                 FBoxLo.z + FClipHi.z * (FBoxHi.z - FBoxLo.z));
  Result := True;
end;

procedure TI2MView.BuildSurface;
var
  S: TI2MSoup;
  L, H: TMcxVec3;
  i: Integer;
  r, g, b: Single;
begin
  FSurf.Clear;
  if FMesh = nil then Exit;
  ClipBox(L, H);
  { a full box reaches past the mesh, so no tet is lost to rounding }
  if FClipLo.x <= 0 then L.x := -1e30;
  if FClipLo.y <= 0 then L.y := -1e30;
  if FClipLo.z <= 0 then L.z := -1e30;
  if FClipHi.x >= 1 then H.x := 1e30;
  if FClipHi.y >= 1 then H.y := 1e30;
  if FClipHi.z >= 1 then H.z := 1e30;
  S := FMesh.CutOut(P3(L.x, L.y, L.z), P3(H.x, H.y, H.z), FHidden);
  FSurf.Reserve(Length(S.P));
  for i := 0 to High(S.Tag) do
  begin
    I2MLabelColour(S.Tag[i], r, g, b);
    FSurf.Tri(S.P[3 * i], S.P[3 * i + 1], S.P[3 * i + 2], r, g, b, 1);
  end;
end;

procedure TI2MView.DefaultAngles;
var
  a, s: Integer;
  Base: Single;
begin
  { from the front: the display's +y -- anterior when the image is oriented
    (SetOrientation turns it to RAS) }
  a := 1;
  s := 1;
  { a flat scene (a picture, a 2-D mesh): from straight above, +y up }
  if (FBoxHi.z - FBoxLo.z <= 1e-3 * Max(FBoxHi.x - FBoxLo.x, FBoxHi.y - FBoxLo.y)) or
     (HasVolume and (FVolDims[2] = 1)) then
  begin
    FCamera.Azimuth := -Pi / 2;
    FCamera.Elevation := 1.5533;
    Exit;
  end;
  { the eye on that side, 0.67 rad (38 degrees) round from it, 0.5 rad up }
  if a = 0 then Base := IfThen(s > 0, 0, Pi) else Base := IfThen(s > 0, Pi / 2, -Pi / 2);
  FCamera.Azimuth := Base - 0.67;
  FCamera.Elevation := 0.5;
end;

procedure TI2MView.FitView(ALeft, ARight: Integer);
var
  Aspect: Single;
  Pad: Single;
  A, B: TMcxVec3;
begin
  if (ALeft < 0) or (ARight < 0) or (ALeft + ARight > FSurface.Width * 2 div 3) then
  begin
    ALeft := 0;
    ARight := 0;
  end;
  Aspect := 1;
  if FSurface.Height > 0 then Aspect := (FSurface.Width - ALeft - ARight) / FSurface.Height;
  Pad := Max(FBoxHi.x - FBoxLo.x, Max(FBoxHi.y - FBoxLo.y, FBoxHi.z - FBoxLo.z)) * 0.05;
  { (the frame on the display) }
  A := ToDisplay(FBoxLo);
  B := ToDisplay(FBoxHi);
  FCamera.FrameBox(McxVec3(Min(A.x, B.x) - Pad, Min(A.y, B.y) - Pad, Min(A.z, B.z) - Pad),
    McxVec3(Max(A.x, B.x) + Pad, Max(A.y, B.y) + Pad, Max(A.z, B.z) + Pad), 45, Aspect);
  { centred in what the margins leave: shifted by half their difference (a
    pixel is 2 d tan(22.5 deg) / height mm at the target) }
  FPanX := 0;
  if FSurface.Height > 0 then
    FPanX := (ALeft - ARight) / 2 * 2 * FCamera.Distance * Tan(22.5 * Pi / 180) / FSurface.Height;
  FPanY := 0;
  Refresh;
end;

procedure TI2MView.Redraw;
begin
  Refresh;
end;

procedure TI2MView.DrawVolume(const AMVP: TMcxMat4);
var
  Eye, Centre, Scale: TMcxVec3;
begin
  if not HasVolume or (FVolShader = nil) or not FShowVolume then Exit;
  Scale := McxVec3(FVolDims[0] * FVoxel.x, FVolDims[1] * FVoxel.y, FVolDims[2] * FVoxel.z);
  Eye := ViewEye;
  Centre := McxVec3(Eye.x / Scale.x, Eye.y / Scale.y, Eye.z / Scale.z);
  if FCube = nil then FCube := TMcxCube.Create;
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glDepthMask(GL_FALSE);
  FVolShader.Use;
  FVolShader.SetMat4('uMVP', AMVP);
  FVolShader.SetVec3('uScale', Scale);
  FVolShader.SetVec3('uEye', Centre);
  FVolShader.SetVec3('uMinSlice', FClipLo);
  FVolShader.SetVec3('uMaxSlice', FClipHi);
  FVolShader.SetFloat('uOpacity', FOpacity);
  FVolShader.SetFloat('uFloor', FFloor);
  FVolShader.SetInt('uMap', FMap);
  if FDragging or FCoarse then FVolShader.SetFloat('uSteps', 96)
  else FVolShader.SetFloat('uSteps', 256);
  FVolShader.SetVec2('uClim', FVolLow, FVolHigh);
  FVolShader.SetInt('uStyle', FStyle);
  FVolShader.SetInt('uLog', 0);
  FVolShader.SetInt('uVolume', 0);
  FVolume.Bind(0);
  FCube.Draw;
  glDepthMask(GL_TRUE);
  glDisable(GL_CULL_FACE);
end;

procedure TI2MView.RenderScene(AWidth, AHeight: Integer);
var
  MVP, W: TMcxMat4;
  L, H, E: TMcxVec3;
  Eps: Single;
  k: Integer;
begin
  if AHeight < 1 then AHeight := 1;
  glViewport(0, 0, AWidth, AHeight);
  glClearColor(FBack.x, FBack.y, FBack.z, 1);
  glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  { the scene turned to the display (RAS): column k, scene axis k's direction }
  W := McxMat4Identity;
  for k := 0 to 2 do
  begin
    W[k * 4 + k] := 0;
    W[k * 4 + FAx[k]] := FSg[k];
  end;
  MVP := McxMat4Mul(McxMat4Perspective(45, AWidth / AHeight, FCamera.Distance * 0.01,
    FCamera.Distance * 10), McxMat4Mul(McxMat4Mul(McxMat4Translate(FPanX, FPanY, 0), FCamera.View), W));
  ClipBox(L, H);
  Eps := 1e-3 * Max(FBoxHi.x - FBoxLo.x, Max(FBoxHi.y - FBoxLo.y, FBoxHi.z - FBoxLo.z));

  FLineShader.Use;
  FLineShader.SetFloat('uEps', Eps);
  FLineShader.SetMat4('uMVP', MVP);
  FLineShader.SetVec3('uLo', McxVec3(-1e30, -1e30, -1e30));
  FLineShader.SetVec3('uHi', McxVec3(1e30, 1e30, 1e30));
  FLineShader.SetInt('uSkip', -1);
  FLineShader.SetFloat('uAlpha', 1);
  if HasContent then   { (nothing open: no frame round a placeholder box) }
  begin
    FFrame.Draw;
    BuildAxisLabels;
    FAxisText.Draw;
  end;

  if FShowMesh and (FMesh <> nil) and (FSurf.Count > 0) then
  begin
    FSolidShader.Use;
    FSolidShader.SetMat4('uMVP', MVP);
    FSolidShader.SetVec3('uLight', McxVec3Norm(ToScene(McxVec3Sub(FCamera.Eye, FCamera.Target))));
    FSolidShader.SetFloat('uAlpha', FMeshAlpha);
    FSolidShader.SetInt('uWire', 0);
    { the faces pushed back a little, so the edges drawn on them win }
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);
    if FMeshAlpha >= 0.99 then
      FSurf.Draw
    else
    begin   { translucent: both sides, far first, no depth writes }
      glDepthMask(GL_FALSE);
      glEnable(GL_CULL_FACE);
      glCullFace(GL_FRONT);
      FSurf.Draw;
      glCullFace(GL_BACK);
      FSurf.Draw;
      glDisable(GL_CULL_FACE);
      glDepthMask(GL_TRUE);
    end;
    glDisable(GL_POLYGON_OFFSET_FILL);
    if FShowEdges then
    begin
      FSolidShader.SetInt('uWire', 1);
      FSolidShader.SetFloat('uWireAlpha', IfThen(FMeshAlpha >= 0.99, 0.9, 0.35 + 0.5 * FMeshAlpha));
      glDepthMask(GL_FALSE);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      FSurf.Draw;
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glDepthMask(GL_TRUE);
    end;
  end;

  if FHasShapes and FShowShapes and (Length(FShapeTris) > 0) then
  begin   { the shape constructs: translucent (their own alphas), both sides, far to near }
    E := ViewEye;
    if not FShapeSorted or (Sqr(E.x - FShapeEye.x) + Sqr(E.y - FShapeEye.y) + Sqr(E.z - FShapeEye.z) > 1e-8) then
      SortShapes(E);
    FSolidShader.Use;
    FSolidShader.SetMat4('uMVP', MVP);
    FSolidShader.SetVec3('uLight', McxVec3Norm(ToScene(McxVec3Sub(FCamera.Eye, FCamera.Target))));
    FSolidShader.SetFloat('uAlpha', 1);
    FSolidShader.SetInt('uWire', 0);
    glDepthMask(GL_FALSE);   { (still tested: a mesh or an image in front hides them) }
    FShapeSurf.Draw;
    glDepthMask(GL_TRUE);
  end;

  DrawVolume(MVP);
end;

procedure TI2MView.GLPaint(Sender: TObject);
begin
  if not FGL.MakeCurrent then Exit;
  if not Start then
  begin
    glClearColor(FBack.x, FBack.y, FBack.z, 1);
    glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
    FGL.SwapBuffers;
    Exit;
  end;
  FPainting := True;
  try
    RenderScene(FGL.Width, FGL.Height);
    FGL.SwapBuffers;
    if FCoarse then
    begin
      FCoarse := False;
      Refresh;   { (queued: after this paint) }
    end;
  finally
    FPainting := False;
  end;
end;

procedure TI2MView.GLMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  { left: orbit; ctrl + left, right or middle: pan }
  FPanning := (Button = mbRight) or (Button = mbMiddle) or ((Button = mbLeft) and (ssCtrl in Shift));
  FDragging := (Button = mbLeft) and not FPanning;
  FDragX := X;
  FDragY := Y;
end;

procedure TI2MView.GLMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
var
  s: Single;
begin
  if FDragging then
    FCamera.Orbit((FDragX - X) * 0.008, (Y - FDragY) * 0.008)
  else if FPanning then
  begin   { shift the view in the screen plane (the orbit's centre stays) }
    s := FCamera.Distance * 0.0015;
    FPanX := FPanX + (X - FDragX) * s;
    FPanY := FPanY - (Y - FDragY) * s;
  end
  else
    Exit;
  FDragX := X;
  FDragY := Y;
  Refresh;
end;

procedure TI2MView.GLMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
begin
  FDragging := False;
  FPanning := False;
  Refresh;
end;

procedure TI2MView.GLMouseWheel(Sender: TObject; Shift: TShiftState; WheelDelta: Integer;
  MousePos: TPoint; var Handled: Boolean);
begin
  FCamera.Zoom(WheelDelta / 120);
  FCoarse := True;
  Refresh;
  Handled := True;
end;

function TI2MView.SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
var
  T: TMcxTarget;
  Px: TBytes;
  Img: TFPMemoryImage;
  W: TFPWriterPNG;
  x, y, o: Integer;
  C: TFPColor;
begin
  Result := False;
  if not Current or not Start then Exit;
  T := TMcxTarget.Create;
  try
    if not T.Bind(AWidth, AHeight) then Exit;
    RenderScene(AWidth, AHeight);
    Px := T.ReadAll;
    T.Unbind;
    Img := TFPMemoryImage.Create(AWidth, AHeight);
    W := TFPWriterPNG.Create;
    try
      for y := 0 to AHeight - 1 do
        for x := 0 to AWidth - 1 do
        begin
          o := 3 * ((AHeight - 1 - y) * AWidth + x);
          C.red := Px[o] * 257;
          C.green := Px[o + 1] * 257;
          C.blue := Px[o + 2] * 257;
          C.alpha := $FFFF;
          Img.Colors[x, y] := C;
        end;
      Img.SaveToFile(AFileName, W);
      Result := True;
    finally
      W.Free;
      Img.Free;
    end;
  finally
    T.Free;
  end;
  Refresh;
end;

end.
