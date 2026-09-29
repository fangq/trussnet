{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mmesh -- a v2mesh tetrahedral mesh (.jmsh / .bmsh), and what the
  preview draws of it: a cut-out. Or a surface mesh (MeshTri / MeshSurf of a
  .jmsh / .bmsh, .off, .stl): its triangles with centroids in the box.

  The tets whose centroids are inside the x/y/z box (and whose labels are
  shown) are kept, and the faces drawn are those between a kept tet and one
  that is not (or none: the exterior), plus the region interfaces between two
  kept tets of different labels, owned by the higher label (usually the
  inclusion) -- iso2mesh's volface of the selection, as mcxcloud's mesh.js
  (same author) computes it. With the whole box that is the mesh's surface;
  a smaller box shows the tetrahedra themselves where it cuts.

  Every unique face and its one or two tets are found once, at load, so a new
  box is one pass over the tets and one over the faces.

  No LCL and no GL in here. }
unit i2mmesh;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, mcxjd, i2mvol;

type
  TI2MPoint = record
    x, y, z: Single;
  end;

  { A triangle soup: 3 points per triangle, the label of its tet. }
  TI2MSoup = record
    P: array of TI2MPoint;
    Tag: array of Integer;
  end;

  TI2MLabels = array of Integer;
  TI2MValues = array of Single;

  TI2MMesh = class
  private
    FNodes: array of TI2MPoint;   { display coordinates }
    FElems: array of array[0..3] of Integer;   { 0-based }
    FTags: array of Integer;
    { every unique face: its tets, as 4 * tet + local face; FFaceB -1 for an
      exterior face }
    FFaceA, FFaceB: array of LongInt;
    FSurface: Integer;   { faces drawn with the whole box }
    { a surface mesh (no tets): triangles, and the labels on their two sides
      (v2mesh's MeshTri [v1 v2 v3 inner outer]; 0 the exterior) }
    FTris: array of array[0..2] of Integer;
    FTriIn, FTriOut: array of Integer;
    FLo, FHi: TI2MPoint;
    FMaxTag: Integer;
    FError: string;
    procedure BuildSurface;
    procedure UpdateBounds;
    function LoadOff(const AFileName: string): Boolean;
    function LoadStl(const AFileName: string): Boolean;
    procedure SetTris(const AArr: TMcxArray; nn: Integer);
    function CutTris(const ALo, AHi: TI2MPoint; const AHidden: array of Boolean): TI2MSoup;
  public
    { Reads MeshNode / MeshElem ([n,3] / [m,4 or 5], 1-based), else MeshTri /
      MeshSurf ([m,3], [m,4]: + a label, [m,5]: + inner, outer); or .off / .stl. }
    function LoadFromFile(const AFileName: string): Boolean;
    { World -> display: p' = (inv(affine) * p + AShift) * AScale, per axis:
      an image's voxels (voxel centres at i + 0.5) in millimetres. }
    procedure ToDisplay(const AAffine: TI2MAffine; AShift: Single; const AScale: TI2MPoint);
    { The cut-out of the tets with centroids in [ALo, AHi] and a label that
      AHidden does not flag (AHidden[tag]; tags past its end are shown). }
    function CutOut(const ALo, AHi: TI2MPoint; const AHidden: array of Boolean): TI2MSoup;
    { The labels present, ascending. }
    function Labels: TI2MLabels;
    { Per element shown (a tet, or a surface triangle, with a label AHidden does
      not flag): its shape quality, 1 for a regular one -- a tet's
      12 (3V)^(2/3) / sum l^2 (Joe-Liu), a triangle's 4 sqrt(3) A / sum l^2 --
      and its size, the volume (a triangle: the area) in display units. }
    procedure ElementStats(const AHidden: array of Boolean; out AQuality, ASize: TI2MValues);
    function NodeCount: Integer;
    function ElemCount: Integer;
    function FaceCount: Integer;
    function IsSurface: Boolean;
    property Lo: TI2MPoint read FLo;
    property Hi: TI2MPoint read FHi;
    property MaxTag: Integer read FMaxTag;
    property Error: string read FError;
  end;

implementation

const
  { the 4 faces of a tet, wound outward for positive orientation (mesh.js) }
  TetFaces: array[0..3, 0..2] of Integer = ((0, 2, 1), (1, 3, 0), (0, 3, 2), (1, 2, 3));

type
  TFaceKey = record
    Key: QWord;   { the sorted node triple, 21 bits each }
    Elem: Integer;
    Local: Integer;
  end;
  TFaceKeys = array of TFaceKey;

procedure SortKeys(var A: TFaceKeys; L, R: Integer);
var
  i, j: Integer;
  p: QWord;
  t: TFaceKey;
begin
  while R - L > 16 do
  begin
    p := A[(L + R) shr 1].Key;
    i := L;
    j := R;
    repeat
      while A[i].Key < p do Inc(i);
      while A[j].Key > p do Dec(j);
      if i <= j then
      begin
        t := A[i];
        A[i] := A[j];
        A[j] := t;
        Inc(i);
        Dec(j);
      end;
    until i > j;
    if j - L < R - i then
    begin
      SortKeys(A, L, j);
      L := i;
    end
    else
    begin
      SortKeys(A, i, R);
      R := j;
    end;
  end;
  for i := L + 1 to R do
  begin
    t := A[i];
    j := i - 1;
    while (j >= L) and (A[j].Key > t.Key) do
    begin
      A[j + 1] := A[j];
      Dec(j);
    end;
    A[j + 1] := t;
  end;
end;

function TI2MMesh.LoadFromFile(const AFileName: string): Boolean;
var
  Arrs: TMcxArrayList;
  BJ: TMcxBJData;
  nn, ne, cols, i, k, v: Integer;
  Ext: string;
const
  Names: array[0..3] of string = ('MeshNode', 'MeshElem', 'MeshTri', 'MeshSurf');
begin
  Result := False;
  FError := '';
  FElems := nil;
  FTags := nil;
  FTris := nil;
  FTriIn := nil;
  FTriOut := nil;
  FFaceA := nil;
  FFaceB := nil;
  FMaxTag := 0;
  Ext := LowerCase(ExtractFileExt(AFileName));
  try
    if Ext = '.off' then Exit(LoadOff(AFileName));
    if Ext = '.stl' then Exit(LoadStl(AFileName));
    { mcxjd's loader picks BJData by extension and does not know .bmsh }
    if Ext = '.bmsh' then
    begin
      BJ := TMcxBJData.Create;
      try
        SetLength(Arrs, Length(Names));
        if not BJ.LoadFromFile(AFileName) then
        begin
          FError := 'cannot read ' + ExtractFileName(AFileName) + ': ' + BJ.Error;
          Exit;
        end;
        for i := 0 to High(Names) do
          if not BJ.GetArray(Names[i], Arrs[i]) then Arrs[i].Dims := nil;
      finally
        BJ.Free;
      end;
    end
    else if not McxLoadArrays(AFileName, Names, Arrs) then
    begin
      FError := 'cannot read ' + ExtractFileName(AFileName);
      Exit;
    end;
  except
    on E: Exception do
    begin
      FError := E.Message;
      Exit;
    end;
  end;
  if (Length(Arrs) < 4) or (Length(Arrs[0].Dims) < 2) or (Arrs[0].Dims[1] < 3) then
  begin
    FError := 'no MeshNode [n,3] array in ' + ExtractFileName(AFileName);
    Exit;
  end;
  nn := Arrs[0].Dims[0];
  if nn >= 1 shl 21 then
  begin
    FError := Format('%d nodes: more than v2m handles (2097151)', [nn]);
    Exit;
  end;
  SetLength(FNodes, nn);
  for i := 0 to nn - 1 do
  begin
    FNodes[i].x := McxArrayValue(Arrs[0], Int64(i) * Arrs[0].Dims[1]);
    FNodes[i].y := McxArrayValue(Arrs[0], Int64(i) * Arrs[0].Dims[1] + 1);
    FNodes[i].z := McxArrayValue(Arrs[0], Int64(i) * Arrs[0].Dims[1] + 2);
  end;
  if (Length(Arrs[1].Dims) < 2) or (Arrs[1].Dims[1] < 4) then
  begin
    { no tets: a surface }
    for k := 2 to 3 do
      if (Length(Arrs[k].Dims) >= 2) and (Arrs[k].Dims[1] >= 3) then
      begin
        SetTris(Arrs[k], nn);
        if FError <> '' then Exit;
        UpdateBounds;
        Exit(True);
      end;
    FError := 'no MeshElem [m,4 or 5] nor MeshTri [m,3..5] in ' + ExtractFileName(AFileName) +
      ' (a 2-D mesh is not shown)';
    Exit;
  end;
  ne := Arrs[1].Dims[0];
  cols := Arrs[1].Dims[1];
  SetLength(FElems, ne);
  SetLength(FTags, ne);
  for i := 0 to ne - 1 do
  begin
    for k := 0 to 3 do
    begin
      v := Round(McxArrayValue(Arrs[1], Int64(i) * cols + k)) - 1;
      if (v < 0) or (v >= nn) then
      begin
        FError := Format('element %d refers to node %d of %d', [i + 1, v + 1, nn]);
        Exit;
      end;
      FElems[i][k] := v;
    end;
    if cols >= 5 then FTags[i] := Round(McxArrayValue(Arrs[1], Int64(i) * cols + cols - 1))
    else FTags[i] := 1;
    if FTags[i] > FMaxTag then FMaxTag := FTags[i];
  end;
  UpdateBounds;
  BuildSurface;
  Result := True;
end;

procedure TI2MMesh.SetTris(const AArr: TMcxArray; nn: Integer);
var
  nt, cols, i, k, v: Integer;
begin
  nt := AArr.Dims[0];
  cols := AArr.Dims[1];
  SetLength(FTris, nt);
  SetLength(FTriIn, nt);
  SetLength(FTriOut, nt);
  for i := 0 to nt - 1 do
  begin
    for k := 0 to 2 do
    begin
      v := Round(McxArrayValue(AArr, Int64(i) * cols + k)) - 1;
      if (v < 0) or (v >= nn) then
      begin
        FError := Format('triangle %d refers to node %d of %d', [i + 1, v + 1, nn]);
        Exit;
      end;
      FTris[i][k] := v;
    end;
    FTriIn[i] := 1;
    FTriOut[i] := 0;
    if cols >= 4 then FTriIn[i] := Round(McxArrayValue(AArr, Int64(i) * cols + 3));
    if cols >= 5 then FTriOut[i] := Round(McxArrayValue(AArr, Int64(i) * cols + 4));
    FMaxTag := Max(FMaxTag, Max(FTriIn[i], FTriOut[i]));
  end;
  FSurface := nt;
end;

{ the whitespace-separated words of a text file, '#' comments dropped }
function Words(const AFileName: string): TStringArray;
var
  L: TStringList;
  i, n, p: Integer;
  s, w: string;
  Parts: TStringArray;
begin
  Result := nil;
  L := TStringList.Create;
  try
    L.LoadFromFile(AFileName);
    n := 0;
    for i := 0 to L.Count - 1 do
    begin
      s := L[i];
      p := Pos('#', s);
      if p > 0 then SetLength(s, p - 1);
      Parts := s.Split([' ', #9, #13], TStringSplitOptions.ExcludeEmpty);
      if n + Length(Parts) > Length(Result) then SetLength(Result, 2 * (n + Length(Parts)) + 16);
      for w in Parts do
      begin
        Result[n] := w;
        Inc(n);
      end;
    end;
    SetLength(Result, n);
  finally
    L.Free;
  end;
end;

function TI2MMesh.LoadOff(const AFileName: string): Boolean;
var
  W: TStringArray;
  p, nv, nf, i, k, c, a, b, d: Integer;
  T: array of array[0..2] of Integer;
  nt: Integer;
begin
  Result := False;
  W := Words(AFileName);
  p := 0;
  if (Length(W) > 0) and (UpperCase(W[0]).EndsWith('OFF')) then Inc(p);
  if Length(W) < p + 3 then
  begin
    FError := 'not an OFF file: ' + ExtractFileName(AFileName);
    Exit;
  end;
  nv := StrToIntDef(W[p], -1);
  nf := StrToIntDef(W[p + 1], -1);
  Inc(p, 3);
  if (nv < 0) or (nf < 0) or (Length(W) < p + 3 * nv) then
  begin
    FError := 'a truncated OFF file: ' + ExtractFileName(AFileName);
    Exit;
  end;
  if nv >= 1 shl 21 then
  begin
    FError := Format('%d nodes: more than v2m handles (2097151)', [nv]);
    Exit;
  end;
  SetLength(FNodes, nv);
  for i := 0 to nv - 1 do
  begin
    FNodes[i].x := StrToFloatDef(W[p], 0, DefaultFormatSettings);
    FNodes[i].y := StrToFloatDef(W[p + 1], 0, DefaultFormatSettings);
    FNodes[i].z := StrToFloatDef(W[p + 2], 0, DefaultFormatSettings);
    Inc(p, 3);
  end;
  { polygons: fans of triangles }
  T := nil;
  nt := 0;
  for i := 0 to nf - 1 do
  begin
    if p >= Length(W) then Break;
    c := StrToIntDef(W[p], 0);
    if (c < 3) or (p + c >= Length(W)) then Break;
    a := StrToIntDef(W[p + 1], -1);
    for k := 2 to c - 1 do
    begin
      b := StrToIntDef(W[p + k], -1);
      d := StrToIntDef(W[p + k + 1], -1);
      if (a < 0) or (b < 0) or (d < 0) or (a >= nv) or (b >= nv) or (d >= nv) then
      begin
        FError := Format('face %d refers to a node past %d', [i + 1, nv]);
        Exit;
      end;
      if nt >= Length(T) then SetLength(T, 2 * nt + 16);
      T[nt][0] := a;
      T[nt][1] := b;
      T[nt][2] := d;
      Inc(nt);
    end;
    Inc(p, c + 1);
  end;
  SetLength(T, nt);
  FTris := T;
  SetLength(FTriIn, nt);
  SetLength(FTriOut, nt);
  for i := 0 to nt - 1 do
  begin
    FTriIn[i] := 1;
    FTriOut[i] := 0;
  end;
  FMaxTag := 1;
  FSurface := nt;
  UpdateBounds;
  Result := nt > 0;
  if not Result then FError := 'no faces in ' + ExtractFileName(AFileName);
end;

function TI2MMesh.LoadStl(const AFileName: string): Boolean;
var
  F: TFileStream;
  Hdr: array[0..79] of Byte;
  n: LongWord;
  Rec: packed record
    N, A, B, C: array[0..2] of Single;
    Attr: Word;
  end;
  i, k, nt: Integer;
  W: TStringArray;
begin
  Result := False;
  nt := 0;
  F := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyWrite);
  try
    n := 0;
    { binary: an 80-byte header, a count, 50 bytes a triangle }
    if (F.Size >= 84) and (F.Read(Hdr, 80) = 80) and (F.Read(n, 4) = 4) and
       (F.Size = 84 + Int64(n) * 50) then
    begin
      SetLength(FNodes, 3 * n);
      for i := 0 to n - 1 do
      begin
        F.ReadBuffer(Rec, 50);
        FNodes[3 * i].x := Rec.A[0]; FNodes[3 * i].y := Rec.A[1]; FNodes[3 * i].z := Rec.A[2];
        FNodes[3 * i + 1].x := Rec.B[0]; FNodes[3 * i + 1].y := Rec.B[1]; FNodes[3 * i + 1].z := Rec.B[2];
        FNodes[3 * i + 2].x := Rec.C[0]; FNodes[3 * i + 2].y := Rec.C[1]; FNodes[3 * i + 2].z := Rec.C[2];
      end;
      nt := n;
    end;
  finally
    F.Free;
  end;
  if nt = 0 then
  begin
    { ASCII: every 'vertex x y z', three a facet }
    W := Words(AFileName);
    SetLength(FNodes, 0);
    k := 0;
    for i := 0 to High(W) - 3 do
      if LowerCase(W[i]) = 'vertex' then
      begin
        if k >= Length(FNodes) then SetLength(FNodes, 2 * k + 48);
        FNodes[k].x := StrToFloatDef(W[i + 1], 0, DefaultFormatSettings);
        FNodes[k].y := StrToFloatDef(W[i + 2], 0, DefaultFormatSettings);
        FNodes[k].z := StrToFloatDef(W[i + 3], 0, DefaultFormatSettings);
        Inc(k);
      end;
    nt := k div 3;
    SetLength(FNodes, 3 * nt);
  end;
  if nt = 0 then
  begin
    FError := 'no triangles in ' + ExtractFileName(AFileName);
    Exit;
  end;
  if 3 * nt >= 1 shl 21 then
  begin
    FError := Format('%d triangles: more than v2m handles', [nt]);
    Exit;
  end;
  { (the corners are not welded: each triangle has its own three nodes) }
  SetLength(FTris, nt);
  SetLength(FTriIn, nt);
  SetLength(FTriOut, nt);
  for i := 0 to nt - 1 do
  begin
    FTris[i][0] := 3 * i;
    FTris[i][1] := 3 * i + 1;
    FTris[i][2] := 3 * i + 2;
    FTriIn[i] := 1;
    FTriOut[i] := 0;
  end;
  FMaxTag := 1;
  FSurface := nt;
  UpdateBounds;
  Result := True;
end;

procedure TI2MMesh.UpdateBounds;
var
  i: Integer;
begin
  if Length(FNodes) = 0 then Exit;
  FLo := FNodes[0];
  FHi := FNodes[0];
  for i := 1 to High(FNodes) do
  begin
    FLo.x := Min(FLo.x, FNodes[i].x); FHi.x := Max(FHi.x, FNodes[i].x);
    FLo.y := Min(FLo.y, FNodes[i].y); FHi.y := Max(FHi.y, FNodes[i].y);
    FLo.z := Min(FLo.z, FNodes[i].z); FHi.z := Max(FHi.z, FNodes[i].z);
  end;
end;

procedure TI2MMesh.ToDisplay(const AAffine: TI2MAffine; AShift: Single; const AScale: TI2MPoint);
var
  Inv: TI2MAffine;
  i: Integer;
  x, y, z: Double;
begin
  if not I2MInvert(AAffine, Inv) then Exit;
  for i := 0 to High(FNodes) do
  begin
    x := FNodes[i].x;
    y := FNodes[i].y;
    z := FNodes[i].z;
    FNodes[i].x := (Inv[0] * x + Inv[1] * y + Inv[2] * z + Inv[3] + AShift) * AScale.x;
    FNodes[i].y := (Inv[4] * x + Inv[5] * y + Inv[6] * z + Inv[7] + AShift) * AScale.y;
    FNodes[i].z := (Inv[8] * x + Inv[9] * y + Inv[10] * z + Inv[11] + AShift) * AScale.z;
  end;
  UpdateBounds;
end;

procedure TI2MMesh.BuildSurface;
var
  K: TFaceKeys;
  ne, e, f, a, b, c, nlo, nhi, md, i, j, n: Integer;
begin
  ne := Length(FElems);
  SetLength(K, 4 * ne);
  for e := 0 to ne - 1 do
    for f := 0 to 3 do
    begin
      a := FElems[e][TetFaces[f, 0]];
      b := FElems[e][TetFaces[f, 1]];
      c := FElems[e][TetFaces[f, 2]];
      nlo := Min(a, Min(b, c));
      nhi := Max(a, Max(b, c));
      md := a + b + c - nlo - nhi;
      K[4 * e + f].Key := (QWord(nlo) shl 42) or (QWord(md) shl 21) or QWord(nhi);
      K[4 * e + f].Elem := e;
      K[4 * e + f].Local := f;
    end;
  if Length(K) > 1 then SortKeys(K, 0, High(K));
  SetLength(FFaceA, Length(K));
  SetLength(FFaceB, Length(K));
  n := 0;
  FSurface := 0;
  i := 0;
  while i < Length(K) do
  begin
    j := i + 1;
    while (j < Length(K)) and (K[j].Key = K[i].Key) do Inc(j);
    FFaceA[n] := 4 * K[i].Elem + K[i].Local;
    if j - i >= 2 then
    begin
      FFaceB[n] := 4 * K[i + 1].Elem + K[i + 1].Local;
      if FTags[K[i].Elem] <> FTags[K[i + 1].Elem] then Inc(FSurface);
    end
    else
    begin
      FFaceB[n] := -1;
      Inc(FSurface);
    end;
    Inc(n);
    i := j;
  end;
  SetLength(FFaceA, n);
  SetLength(FFaceB, n);
end;

function TI2MMesh.CutOut(const ALo, AHi: TI2MPoint; const AHidden: array of Boolean): TI2MSoup;
var
  Keep: array of Boolean;
  e, i, n, own, t, ta, tb: Integer;
  cx, cy, cz: Single;
  ka, kb: Boolean;
  All: Boolean;
begin
  if FTris <> nil then Exit(CutTris(ALo, AHi, AHidden));
  SetLength(Keep, Length(FElems));
  All := (ALo.x <= FLo.x) and (ALo.y <= FLo.y) and (ALo.z <= FLo.z) and
         (AHi.x >= FHi.x) and (AHi.y >= FHi.y) and (AHi.z >= FHi.z);
  for e := 0 to High(FElems) do
  begin
    t := FTags[e];
    if (t >= 0) and (t <= High(AHidden)) and AHidden[t] then
    begin
      Keep[e] := False;
      Continue;
    end;
    if All then
    begin
      Keep[e] := True;
      Continue;
    end;
    cx := (FNodes[FElems[e][0]].x + FNodes[FElems[e][1]].x + FNodes[FElems[e][2]].x + FNodes[FElems[e][3]].x) * 0.25;
    cy := (FNodes[FElems[e][0]].y + FNodes[FElems[e][1]].y + FNodes[FElems[e][2]].y + FNodes[FElems[e][3]].y) * 0.25;
    cz := (FNodes[FElems[e][0]].z + FNodes[FElems[e][1]].z + FNodes[FElems[e][2]].z + FNodes[FElems[e][3]].z) * 0.25;
    Keep[e] := (cx >= ALo.x) and (cx <= AHi.x) and (cy >= ALo.y) and (cy <= AHi.y) and
               (cz >= ALo.z) and (cz <= AHi.z);
  end;
  { two passes: count, then fill }
  SetLength(Result.P, 0);
  SetLength(Result.Tag, 0);
  for n := 0 to 1 do
  begin
    t := 0;
    for i := 0 to High(FFaceA) do
    begin
      ka := Keep[FFaceA[i] shr 2];
      kb := (FFaceB[i] >= 0) and Keep[FFaceB[i] shr 2];
      if ka and not kb then own := FFaceA[i]
      else if kb and not ka then own := FFaceB[i]
      else if ka and kb then
      begin
        ta := FTags[FFaceA[i] shr 2];
        tb := FTags[FFaceB[i] shr 2];
        if ta = tb then Continue;
        if ta > tb then own := FFaceA[i] else own := FFaceB[i];
      end
      else
        Continue;
      if n = 1 then
      begin
        e := own shr 2;
        Result.P[3 * t] := FNodes[FElems[e][TetFaces[own and 3, 0]]];
        Result.P[3 * t + 1] := FNodes[FElems[e][TetFaces[own and 3, 1]]];
        Result.P[3 * t + 2] := FNodes[FElems[e][TetFaces[own and 3, 2]]];
        Result.Tag[t] := FTags[e];
      end;
      Inc(t);
    end;
    if n = 0 then
    begin
      SetLength(Result.P, 3 * t);
      SetLength(Result.Tag, t);
    end;
  end;
end;

{ a label shown: not flagged in AHidden (tags past its end are shown) }
function Shown(t: Integer; const AHidden: array of Boolean): Boolean;
begin
  Result := not ((t >= 0) and (t <= High(AHidden)) and AHidden[t]);
end;

function TI2MMesh.CutTris(const ALo, AHi: TI2MPoint; const AHidden: array of Boolean): TI2MSoup;
var
  i, n, t, k: Integer;
  c: TI2MPoint;
  Keep: array of Boolean;
begin
  SetLength(Keep, Length(FTris));
  n := 0;
  for i := 0 to High(FTris) do
  begin
    { a face bounds each of its two regions: shown while either one is }
    Keep[i] := ((FTriIn[i] > 0) and Shown(FTriIn[i], AHidden)) or ((FTriOut[i] > 0) and Shown(FTriOut[i], AHidden));
    if not Keep[i] then Continue;
    c.x := 0; c.y := 0; c.z := 0;
    for k := 0 to 2 do
    begin
      c.x := c.x + FNodes[FTris[i][k]].x / 3;
      c.y := c.y + FNodes[FTris[i][k]].y / 3;
      c.z := c.z + FNodes[FTris[i][k]].z / 3;
    end;
    Keep[i] := (c.x >= ALo.x) and (c.x <= AHi.x) and (c.y >= ALo.y) and (c.y <= AHi.y) and
               (c.z >= ALo.z) and (c.z <= AHi.z);
    if Keep[i] then Inc(n);
  end;
  SetLength(Result.P, 3 * n);
  SetLength(Result.Tag, n);
  t := 0;
  for i := 0 to High(FTris) do
    if Keep[i] then
    begin
      for k := 0 to 2 do Result.P[3 * t + k] := FNodes[FTris[i][k]];
      Result.Tag[t] := Max(FTriIn[i], FTriOut[i]);   { the inclusion's colour, as the tets' }
      Inc(t);
    end;
end;

function TI2MMesh.Labels: TI2MLabels;
var
  Seen: array of Boolean;
  e, t, n: Integer;
begin
  Result := nil;
  if FMaxTag < 0 then Exit;
  SetLength(Seen, FMaxTag + 1);
  for e := 0 to High(FTags) do
    if FTags[e] >= 0 then Seen[FTags[e]] := True;
  for e := 0 to High(FTriIn) do
  begin
    if FTriIn[e] > 0 then Seen[FTriIn[e]] := True;
    if FTriOut[e] > 0 then Seen[FTriOut[e]] := True;
  end;
  n := 0;
  for t := 0 to FMaxTag do
    if Seen[t] then Inc(n);
  SetLength(Result, n);
  n := 0;
  for t := 0 to FMaxTag do
    if Seen[t] then
    begin
      Result[n] := t;
      Inc(n);
    end;
end;

function TI2MMesh.NodeCount: Integer;
begin
  Result := Length(FNodes);
end;

function TI2MMesh.ElemCount: Integer;
begin
  Result := Length(FElems);
end;

function TI2MMesh.FaceCount: Integer;
begin
  Result := FSurface;
end;

function TI2MMesh.IsSurface: Boolean;
begin
  Result := FTris <> nil;
end;

procedure TI2MMesh.ElementStats(const AHidden: array of Boolean; out AQuality, ASize: TI2MValues);

  function Shown(ATag: Integer): Boolean;
  begin
    Result := (ATag > 0) and ((ATag > High(AHidden)) or not AHidden[ATag]);
  end;

  function D2(const P, Q: TI2MPoint): Double;
  begin
    Result := Sqr(Double(P.x) - Q.x) + Sqr(Double(P.y) - Q.y) + Sqr(Double(P.z) - Q.z);
  end;

var
  i, n: Integer;
  a, b, c, d: TI2MPoint;
  ux, uy, uz, vx, vy, vz, wx, wy, wz, cx, cy, cz, v, ar, l2: Double;
begin
  AQuality := nil;
  ASize := nil;
  n := 0;
  if IsSurface then
  begin
    SetLength(AQuality, Length(FTris));
    SetLength(ASize, Length(FTris));
    for i := 0 to High(FTris) do
    begin
      { (an unlabelled surface: every triangle) }
      if ((FTriIn <> nil) or (FTriOut <> nil)) and not ((FTriIn <> nil) and Shown(FTriIn[i])) and
         not ((FTriOut <> nil) and Shown(FTriOut[i])) then Continue;
      a := FNodes[FTris[i][0]];
      b := FNodes[FTris[i][1]];
      c := FNodes[FTris[i][2]];
      ux := Double(b.x) - a.x; uy := Double(b.y) - a.y; uz := Double(b.z) - a.z;
      vx := Double(c.x) - a.x; vy := Double(c.y) - a.y; vz := Double(c.z) - a.z;
      cx := uy * vz - uz * vy; cy := uz * vx - ux * vz; cz := ux * vy - uy * vx;
      ar := 0.5 * Sqrt(cx * cx + cy * cy + cz * cz);
      l2 := D2(a, b) + D2(b, c) + D2(c, a);
      if l2 > 0 then AQuality[n] := 4 * Sqrt(3) * ar / l2 else AQuality[n] := 0;
      ASize[n] := ar;
      Inc(n);
    end;
  end
  else
  begin
    SetLength(AQuality, Length(FElems));
    SetLength(ASize, Length(FElems));
    for i := 0 to High(FElems) do
    begin
      if (FTags <> nil) and not Shown(FTags[i]) then Continue;
      a := FNodes[FElems[i][0]];
      b := FNodes[FElems[i][1]];
      c := FNodes[FElems[i][2]];
      d := FNodes[FElems[i][3]];
      ux := Double(b.x) - a.x; uy := Double(b.y) - a.y; uz := Double(b.z) - a.z;
      vx := Double(c.x) - a.x; vy := Double(c.y) - a.y; vz := Double(c.z) - a.z;
      wx := Double(d.x) - a.x; wy := Double(d.y) - a.y; wz := Double(d.z) - a.z;
      v := Abs(ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx)) / 6;
      l2 := D2(a, b) + D2(a, c) + D2(a, d) + D2(b, c) + D2(b, d) + D2(c, d);
      if l2 > 0 then AQuality[n] := 12 * Power(3 * v, 2 / 3) / l2 else AQuality[n] := 0;
      ASize[n] := v;
      Inc(n);
    end;
  end;
  SetLength(AQuality, n);
  SetLength(ASize, n);
end;

end.
