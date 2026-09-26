{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mmesh -- a trussnet tetrahedral mesh (.jmsh / .bmsh), and the two things
  the preview draws of it, ported from mcxcloud's mesh.js (same author), which
  follows iso2mesh:

    volface   the renderable surface: the exterior faces (used by one tet) and
              the region interfaces (shared by two tets of different labels),
              each face owned by its higher-labelled tet (usually the inclusion)
    qmeshcut  the exact cross-section of the mesh by an axis-aligned plane:
              every tet straddling it gives a triangle or a quad, interpolated
              along its cut edges

  The surface is clipped to the x/y/z box on the GPU and the cross-sections
  fill the box faces, so the interior regions show where the box cuts them.

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

  TI2MMesh = class
  private
    FNodes: array of TI2MPoint;   { display coordinates }
    FElems: array of array[0..3] of Integer;   { 0-based }
    FTags: array of Integer;
    FFaces: array of array[0..2] of Integer;   { the volface surface }
    FOwner: array of Integer;                   { owning tet per face }
    FLo, FHi: TI2MPoint;
    FMaxTag: Integer;
    FError: string;
    procedure BuildSurface;
    procedure UpdateBounds;
  public
    { Reads MeshNode / MeshElem ([n,3] / [m,4 or 5], 1-based). }
    function LoadFromFile(const AFileName: string): Boolean;
    { World -> display: p' = inv(affine) * p + AShift (voxel space of an image,
      whose texture spans [0, n] with voxel centres at i + 0.5). }
    procedure ToVoxelSpace(const AAffine: TI2MAffine; AShift: Single);
    { The surface faces as a soup (all labels; the caller filters). }
    function Surface: TI2MSoup;
    { The cross-section at axis (0 x, 1 y, 2 z) = APos, display coordinates. }
    function Cut(AAxis: Integer; APos: Single): TI2MSoup;
    function NodeCount: Integer;
    function ElemCount: Integer;
    function FaceCount: Integer;
    property Lo: TI2MPoint read FLo;
    property Hi: TI2MPoint read FHi;
    property MaxTag: Integer read FMaxTag;
    property Error: string read FError;
  end;

implementation

const
  { the 4 faces of a tet, wound outward for positive orientation (mesh.js) }
  TetFaces: array[0..3, 0..2] of Integer = ((0, 2, 1), (1, 3, 0), (0, 3, 2), (1, 2, 3));
  { the 6 edges in combinations(4,2) order: qmeshcut's quad cycle 0,1,3,2
    is only right in this order }
  TetEdges: array[0..5, 0..1] of Integer = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3));

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
begin
  Result := False;
  FError := '';
  try
    { mcxjd's loader picks BJData by extension and does not know .bmsh }
    if LowerCase(ExtractFileExt(AFileName)) = '.bmsh' then
    begin
      BJ := TMcxBJData.Create;
      try
        SetLength(Arrs, 2);
        if not (BJ.LoadFromFile(AFileName) and BJ.GetArray('MeshNode', Arrs[0]) and
                BJ.GetArray('MeshElem', Arrs[1])) then
        begin
          FError := 'cannot read ' + ExtractFileName(AFileName) + ': ' + BJ.Error;
          Exit;
        end;
      finally
        BJ.Free;
      end;
    end
    else if not McxLoadArrays(AFileName, ['MeshNode', 'MeshElem'], Arrs) then
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
  if (Length(Arrs) < 2) or (Length(Arrs[0].Dims) < 2) or (Length(Arrs[1].Dims) < 2) then
  begin
    FError := 'no MeshNode / MeshElem arrays in ' + ExtractFileName(AFileName);
    Exit;
  end;
  nn := Arrs[0].Dims[0];
  ne := Arrs[1].Dims[0];
  cols := Arrs[1].Dims[1];
  if (Arrs[0].Dims[1] < 3) or (cols < 4) then
  begin
    FError := 'MeshNode must be [n,3] and MeshElem [m,4] or [m,5] (a 2-D mesh is not shown)';
    Exit;
  end;
  if nn >= 1 shl 21 then
  begin
    FError := Format('%d nodes: more than img2mesh handles (2097151)', [nn]);
    Exit;
  end;
  SetLength(FNodes, nn);
  for i := 0 to nn - 1 do
  begin
    FNodes[i].x := McxArrayValue(Arrs[0], Int64(i) * Arrs[0].Dims[1]);
    FNodes[i].y := McxArrayValue(Arrs[0], Int64(i) * Arrs[0].Dims[1] + 1);
    FNodes[i].z := McxArrayValue(Arrs[0], Int64(i) * Arrs[0].Dims[1] + 2);
  end;
  SetLength(FElems, ne);
  SetLength(FTags, ne);
  FMaxTag := 0;
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

procedure TI2MMesh.ToVoxelSpace(const AAffine: TI2MAffine; AShift: Single);
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
    FNodes[i].x := Inv[0] * x + Inv[1] * y + Inv[2] * z + Inv[3] + AShift;
    FNodes[i].y := Inv[4] * x + Inv[5] * y + Inv[6] * z + Inv[7] + AShift;
    FNodes[i].z := Inv[8] * x + Inv[9] * y + Inv[10] * z + Inv[11] + AShift;
  end;
  UpdateBounds;
end;

procedure TI2MMesh.BuildSurface;
var
  K: TFaceKeys;
  ne, e, f, a, b, c, nlo, nhi, md, i, j, n, own: Integer;
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
  SetLength(FFaces, 0);
  SetLength(FOwner, 0);
  SetLength(FFaces, Length(K));
  SetLength(FOwner, Length(K));
  n := 0;
  i := 0;
  while i < Length(K) do
  begin
    j := i + 1;
    while (j < Length(K)) and (K[j].Key = K[i].Key) do Inc(j);
    own := -1;
    if j - i = 1 then own := i   { exterior }
    else if (j - i = 2) and (FTags[K[i].Elem] <> FTags[K[i + 1].Elem]) then
    begin   { a region interface: owned by the higher label }
      if FTags[K[i].Elem] > FTags[K[i + 1].Elem] then own := i else own := i + 1;
    end;
    if own >= 0 then
    begin
      e := K[own].Elem;
      f := K[own].Local;
      FFaces[n][0] := FElems[e][TetFaces[f, 0]];
      FFaces[n][1] := FElems[e][TetFaces[f, 1]];
      FFaces[n][2] := FElems[e][TetFaces[f, 2]];
      FOwner[n] := e;
      Inc(n);
    end;
    i := j;
  end;
  SetLength(FFaces, n);
  SetLength(FOwner, n);
end;

function TI2MMesh.Surface: TI2MSoup;
var
  i, k: Integer;
begin
  SetLength(Result.P, 3 * Length(FFaces));
  SetLength(Result.Tag, Length(FFaces));
  for i := 0 to High(FFaces) do
  begin
    for k := 0 to 2 do Result.P[3 * i + k] := FNodes[FFaces[i][k]];
    Result.Tag[i] := FTags[FOwner[i]];
  end;
end;

function Coord(const P: TI2MPoint; AAxis: Integer): Single; inline;
begin
  case AAxis of
    0: Result := P.x;
    1: Result := P.y;
  else
    Result := P.z;
  end;
end;

function TI2MMesh.Cut(AAxis: Integer; APos: Single): TI2MSoup;
var
  e, i, k, ssum, ncut, nt, cap: Integer;
  d: array[0..3] of Single;
  s: array[0..3] of Integer;
  cp: array[0..3] of TI2MPoint;
  t: Single;
  A, B: TI2MPoint;

  procedure Tri(a, b, c: Integer);
  begin
    if 3 * nt + 3 > cap then
    begin
      cap := Max(3 * nt + 3, 2 * cap + 3072);
      SetLength(Result.P, cap);
      SetLength(Result.Tag, cap div 3 + 1);
    end;
    Result.P[3 * nt] := cp[a];
    Result.P[3 * nt + 1] := cp[b];
    Result.P[3 * nt + 2] := cp[c];
    Result.Tag[nt] := FTags[e];
    Inc(nt);
  end;

begin
  nt := 0;
  cap := 0;
  SetLength(Result.P, 0);
  SetLength(Result.Tag, 0);
  for e := 0 to High(FElems) do
  begin
    ssum := 0;
    for i := 0 to 3 do
    begin
      d[i] := Coord(FNodes[FElems[e][i]], AAxis) - APos;
      if d[i] >= 0 then s[i] := 1 else s[i] := -1;
      Inc(ssum, s[i]);
    end;
    if (ssum = 4) or (ssum = -4) then Continue;   { entirely on one side }
    ncut := 0;
    for k := 0 to 5 do
    begin
      if s[TetEdges[k, 0]] + s[TetEdges[k, 1]] <> 0 then Continue;
      A := FNodes[FElems[e][TetEdges[k, 0]]];
      B := FNodes[FElems[e][TetEdges[k, 1]]];
      t := d[TetEdges[k, 0]] / (d[TetEdges[k, 0]] - d[TetEdges[k, 1]]);
      cp[ncut].x := A.x + t * (B.x - A.x);
      cp[ncut].y := A.y + t * (B.y - A.y);
      cp[ncut].z := A.z + t * (B.z - A.z);
      Inc(ncut);
      if ncut = 4 then Break;
    end;
    if ncut = 3 then Tri(0, 1, 2)
    else if ncut = 4 then
    begin   { the quad cycle 0, 1, 3, 2 }
      Tri(0, 1, 3);
      Tri(0, 3, 2);
    end;
  end;
  SetLength(Result.P, 3 * nt);
  SetLength(Result.Tag, nt);
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
  Result := Length(FFaces);
end;

end.
