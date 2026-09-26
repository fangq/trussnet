{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mmesh -- a trussnet tetrahedral mesh (.jmsh / .bmsh), and what the
  preview draws of it: a cut-out.

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

  TI2MMesh = class
  private
    FNodes: array of TI2MPoint;   { display coordinates }
    FElems: array of array[0..3] of Integer;   { 0-based }
    FTags: array of Integer;
    { every unique face: its tets, as 4 * tet + local face; FFaceB -1 for an
      exterior face }
    FFaceA, FFaceB: array of LongInt;
    FSurface: Integer;   { faces drawn with the whole box }
    FLo, FHi: TI2MPoint;
    FMaxTag: Integer;
    FError: string;
    procedure BuildSurface;
    procedure UpdateBounds;
  public
    { Reads MeshNode / MeshElem ([n,3] / [m,4 or 5], 1-based). }
    function LoadFromFile(const AFileName: string): Boolean;
    { World -> display: p' = (inv(affine) * p + AShift) * AScale, per axis:
      an image's voxels (voxel centres at i + 0.5) in millimetres. }
    procedure ToDisplay(const AAffine: TI2MAffine; AShift: Single; const AScale: TI2MPoint);
    { The cut-out of the tets with centroids in [ALo, AHi] and a label that
      AHidden does not flag (AHidden[tag]; tags past its end are shown). }
    function CutOut(const ALo, AHi: TI2MPoint; const AHidden: array of Boolean): TI2MSoup;
    { The labels present, ascending. }
    function Labels: TI2MLabels;
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

end.
