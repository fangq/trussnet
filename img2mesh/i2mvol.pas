{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mvol -- reading the images trussnet reads: NIfTI-1 / NIfTI-2 (.nii,
  .nii.gz) and JNIfTI (.jnii text, .bnii binary), 3-D or 4-D.

  The volume comes back as floats, x fastest (the order the GL texture and
  trussnet's own grid use), channel-major for a 4-D image, with the affine
  that maps a 0-based voxel index (i, j, k) of the file to world coordinates.
  That affine is the one trussnet uses for a file (sform when set, else qform,
  else the voxel size), so a mesh trussnet writes in world coordinates maps
  back onto the voxels with its inverse.

  No LCL and no GL in here. }
unit i2mvol;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, zstream, fpjson, jsonparser, mcxjd;

type
  TI2MAffine = array[0..15] of Double;   { row-major 4x4, voxel -> world }
  TI2MSingles = array of Single;
  TI2MDoubles = array of Double;

  TI2MVolume = record
    Nx, Ny, Nz, Nc: Integer;   { Nc > 1: a 4-D image (e.g. a probability map) }
    Data: array of Single;     { x fastest; channel c at c * Nx * Ny * Nz }
    Affine: TI2MAffine;
    VoxelSize: array[0..2] of Double;
    IsInteger: Boolean;        { every value an integer: a label volume }
    Low, High: Single;
  end;

function I2MLoadVolume(const AFileName: string; out AVol: TI2MVolume;
  out AError: string): Boolean;
{ The inverse of an affine (a general 4x4 inverse; False if singular). }
function I2MInvert(const A: TI2MAffine; out AInv: TI2MAffine): Boolean;
{ The label of the most probable channel per voxel (0-based channel index). }
procedure I2MArgmax(const AVol: TI2MVolume; out ALabels: TI2MSingles);

implementation

procedure SetDiag(var A: TI2MAffine; sx, sy, sz: Double);
var
  i: Integer;
begin
  for i := 0 to 15 do A[i] := 0;
  A[0] := sx;
  A[5] := sy;
  A[10] := sz;
  A[15] := 1;
end;

function I2MInvert(const A: TI2MAffine; out AInv: TI2MAffine): Boolean;
var
  M: array[0..3, 0..7] of Double;
  r, c, p, k: Integer;
  t, f: Double;
begin
  for r := 0 to 3 do
    for c := 0 to 3 do
    begin
      M[r, c] := A[4 * r + c];
      if r = c then M[r, c + 4] := 1 else M[r, c + 4] := 0;
    end;
  for c := 0 to 3 do
  begin
    p := c;
    for r := c + 1 to 3 do
      if Abs(M[r, c]) > Abs(M[p, c]) then p := r;
    if Abs(M[p, c]) < 1e-12 then Exit(False);
    if p <> c then
      for k := 0 to 7 do
      begin
        t := M[c, k];
        M[c, k] := M[p, k];
        M[p, k] := t;
      end;
    f := M[c, c];
    for k := 0 to 7 do M[c, k] := M[c, k] / f;
    for r := 0 to 3 do
      if r <> c then
      begin
        f := M[r, c];
        if f <> 0 then
          for k := 0 to 7 do M[r, k] := M[r, k] - f * M[c, k];
      end;
  end;
  for r := 0 to 3 do
    for c := 0 to 3 do AInv[4 * r + c] := M[r, c + 4];
  Result := True;
end;

procedure I2MArgmax(const AVol: TI2MVolume; out ALabels: TI2MSingles);
var
  nv, v: Int64;
  c, best: Integer;
  bp, q: Single;
begin
  nv := Int64(AVol.Nx) * AVol.Ny * AVol.Nz;
  SetLength(ALabels, nv);
  for v := 0 to nv - 1 do
  begin
    best := 0;
    bp := AVol.Data[v];
    for c := 1 to AVol.Nc - 1 do
    begin
      q := AVol.Data[Int64(c) * nv + v];
      if q > bp then
      begin
        bp := q;
        best := c;
      end;
    end;
    ALabels[v] := best;
  end;
end;

{ ---------------------------------------------------------------- NIfTI --- }

function ReadAllBytes(const AFileName: string; out ABytes: TBytes): Boolean;
var
  F: TFileStream;
  G: TGZFileStream;
  Buf: array[0..65535] of Byte;
  n, len: Integer;
  Head: array[0..1] of Byte;
begin
  Result := False;
  SetLength(ABytes, 0);
  F := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyWrite);
  try
    Head[0] := 0;
    Head[1] := 0;
    if F.Size >= 2 then F.ReadBuffer(Head, 2);
    if not ((Head[0] = $1F) and (Head[1] = $8B)) then
    begin
      F.Position := 0;
      SetLength(ABytes, F.Size);
      if F.Size > 0 then F.ReadBuffer(ABytes[0], F.Size);
      Exit(True);
    end;
  finally
    F.Free;
  end;
  { gzip: inflated in chunks, the size is not known up front }
  G := TGZFileStream.Create(AFileName, gzopenread);
  try
    len := 0;
    repeat
      n := G.Read(Buf, SizeOf(Buf));
      if n > 0 then
      begin
        if len + n > Length(ABytes) then
          SetLength(ABytes, Max(len + n, 2 * Length(ABytes) + 65536));
        Move(Buf[0], ABytes[len], n);
        Inc(len, n);
      end;
    until n <= 0;
    SetLength(ABytes, len);
    Result := True;
  finally
    G.Free;
  end;
end;

function RdI16(const B: TBytes; o: Integer; Swap: Boolean): SmallInt;
var
  w: Word;
begin
  w := B[o] or (Word(B[o + 1]) shl 8);
  if Swap then w := SwapEndian(w);
  Result := SmallInt(w);
end;

function RdI32(const B: TBytes; o: Integer; Swap: Boolean): LongInt;
var
  d: LongWord;
begin
  Move(B[o], d, 4);
  if Swap then d := SwapEndian(d);
  Result := LongInt(d);
end;

function RdI64(const B: TBytes; o: Integer; Swap: Boolean): Int64;
var
  q: QWord;
begin
  Move(B[o], q, 8);
  if Swap then q := SwapEndian(q);
  Result := Int64(q);
end;

function RdF32(const B: TBytes; o: Integer; Swap: Boolean): Double;
var
  d: LongWord;
  s: Single absolute d;
begin
  Move(B[o], d, 4);
  if Swap then d := SwapEndian(d);
  Result := s;
end;

function RdF64(const B: TBytes; o: Integer; Swap: Boolean): Double;
var
  q: QWord;
  s: Double absolute q;
begin
  Move(B[o], q, 8);
  if Swap then q := SwapEndian(q);
  Result := s;
end;

{ The NIfTI quaternion form: rotation from (b, c, d), a = sqrt(1 - b2 - c2 - d2),
  qfac = pixdim[0] (+-1) flips the third axis, then the voxel sizes. }
procedure QuatToAffine(b, c, d, qx, qy, qz, dx, dy, dz, qfac: Double;
  out M: TI2MAffine);
var
  qa, R11, R12, R13, R21, R22, R23, R31, R32, R33: Double;
begin
  qa := 1.0 - (b * b + c * c + d * d);
  if qa < 1e-7 then
  begin
    qa := 1.0 / Sqrt(b * b + c * c + d * d);
    b := b * qa;
    c := c * qa;
    d := d * qa;
    qa := 0;
  end
  else
    qa := Sqrt(qa);
  if qfac < 0 then dz := -dz;
  R11 := qa * qa + b * b - c * c - d * d;
  R12 := 2 * (b * c - qa * d);
  R13 := 2 * (b * d + qa * c);
  R21 := 2 * (b * c + qa * d);
  R22 := qa * qa + c * c - b * b - d * d;
  R23 := 2 * (c * d - qa * b);
  R31 := 2 * (b * d - qa * c);
  R32 := 2 * (c * d + qa * b);
  R33 := qa * qa + d * d - c * c - b * b;
  M[0] := R11 * dx; M[1] := R12 * dy; M[2] := R13 * dz; M[3] := qx;
  M[4] := R21 * dx; M[5] := R22 * dy; M[6] := R23 * dz; M[7] := qy;
  M[8] := R31 * dx; M[9] := R32 * dy; M[10] := R33 * dz; M[11] := qz;
  M[12] := 0; M[13] := 0; M[14] := 0; M[15] := 1;
end;

function LoadNifti(const AFileName: string; out AVol: TI2MVolume;
  out AError: string): Boolean;
var
  B: TBytes;
  Swap, Nii2: Boolean;
  hs, dt, i, qform, sform, esz: Integer;
  dim: array[0..7] of Int64;
  pix: array[0..7] of Double;
  off, n, k: Int64;
  slope, inter, v: Double;
  sx, sy, sz: array[0..3] of Double;
begin
  Result := False;
  AError := '';
  if not ReadAllBytes(AFileName, B) then
  begin
    AError := 'cannot read ' + AFileName;
    Exit;
  end;
  if Length(B) < 348 then
  begin
    AError := 'not a NIfTI file (too short)';
    Exit;
  end;
  Swap := False;
  hs := RdI32(B, 0, False);
  if (hs <> 348) and (hs <> 540) then
  begin
    Swap := True;
    hs := RdI32(B, 0, True);
  end;
  if (hs <> 348) and (hs <> 540) then
  begin
    AError := 'not a NIfTI-1 or NIfTI-2 header';
    Exit;
  end;
  Nii2 := hs = 540;
  if Nii2 and (Length(B) < 540) then
  begin
    AError := 'truncated NIfTI-2 header';
    Exit;
  end;

  if Nii2 then
  begin
    dt := RdI16(B, 12, Swap);
    for i := 0 to 7 do dim[i] := RdI64(B, 16 + 8 * i, Swap);
    for i := 0 to 7 do pix[i] := RdF64(B, 104 + 8 * i, Swap);
    off := RdI64(B, 168, Swap);
    slope := RdF64(B, 176, Swap);
    inter := RdF64(B, 184, Swap);
    qform := RdI32(B, 344, Swap);
    sform := RdI32(B, 348, Swap);
    if sform > 0 then
      for i := 0 to 3 do
      begin
        sx[i] := RdF64(B, 400 + 8 * i, Swap);
        sy[i] := RdF64(B, 432 + 8 * i, Swap);
        sz[i] := RdF64(B, 464 + 8 * i, Swap);
      end
    else if qform > 0 then
      QuatToAffine(RdF64(B, 352, Swap), RdF64(B, 360, Swap), RdF64(B, 368, Swap),
        RdF64(B, 376, Swap), RdF64(B, 384, Swap), RdF64(B, 392, Swap),
        pix[1], pix[2], pix[3], pix[0], AVol.Affine);
  end
  else
  begin
    dt := RdI16(B, 70, Swap);
    for i := 0 to 7 do dim[i] := RdI16(B, 40 + 2 * i, Swap);
    for i := 0 to 7 do pix[i] := RdF32(B, 76 + 4 * i, Swap);
    off := Round(RdF32(B, 108, Swap));
    slope := RdF32(B, 112, Swap);
    inter := RdF32(B, 116, Swap);
    qform := RdI16(B, 252, Swap);
    sform := RdI16(B, 254, Swap);
    if sform > 0 then
      for i := 0 to 3 do
      begin
        sx[i] := RdF32(B, 280 + 4 * i, Swap);
        sy[i] := RdF32(B, 296 + 4 * i, Swap);
        sz[i] := RdF32(B, 312 + 4 * i, Swap);
      end
    else if qform > 0 then
      QuatToAffine(RdF32(B, 256, Swap), RdF32(B, 260, Swap), RdF32(B, 264, Swap),
        RdF32(B, 268, Swap), RdF32(B, 272, Swap), RdF32(B, 276, Swap),
        pix[1], pix[2], pix[3], pix[0], AVol.Affine);
  end;

  if (dim[0] < 3) or (dim[1] < 1) or (dim[2] < 1) or (dim[3] < 1) then
  begin
    AError := Format('a %d-D image: img2mesh wants 3-D or 4-D', [dim[0]]);
    Exit;
  end;
  AVol.Nx := dim[1];
  AVol.Ny := dim[2];
  AVol.Nz := dim[3];
  AVol.Nc := 1;
  if (dim[0] >= 4) and (dim[4] > 1) then AVol.Nc := dim[4];
  for i := 0 to 2 do
  begin
    AVol.VoxelSize[i] := Abs(pix[i + 1]);
    if AVol.VoxelSize[i] <= 0 then AVol.VoxelSize[i] := 1;
  end;
  if sform > 0 then
    for i := 0 to 3 do
    begin
      AVol.Affine[i] := sx[i];
      AVol.Affine[4 + i] := sy[i];
      AVol.Affine[8 + i] := sz[i];
      AVol.Affine[12 + i] := Ord(i = 3);
    end
  else if qform <= 0 then
    SetDiag(AVol.Affine, AVol.VoxelSize[0], AVol.VoxelSize[1], AVol.VoxelSize[2]);

  case dt of
    2, 256: esz := 1;          { uint8, int8 }
    4, 512: esz := 2;          { int16, uint16 }
    8, 16, 768: esz := 4;      { int32, float32, uint32 }
    64, 1024, 1280: esz := 8;  { float64, int64, uint64 }
  else
    begin
      AError := Format('unsupported NIfTI datatype %d', [dt]);
      Exit;
    end;
  end;
  if off < hs then off := hs;
  n := Int64(AVol.Nx) * AVol.Ny * AVol.Nz * AVol.Nc;
  if off + n * esz > Length(B) then
  begin
    AError := 'the NIfTI data is shorter than its header says';
    Exit;
  end;
  if (slope = 0) or IsNan(slope) then
  begin
    slope := 1;
    inter := 0;
  end;
  if IsNan(inter) then inter := 0;

  SetLength(AVol.Data, n);
  AVol.IsInteger := (dt <> 16) and (dt <> 64) and (slope = 1) and (inter = 0);
  for k := 0 to n - 1 do
  begin
    case dt of
      2: v := B[off + k];
      256: v := ShortInt(B[off + k]);
      4: v := RdI16(B, off + 2 * k, Swap);
      512: v := Word(RdI16(B, off + 2 * k, Swap));
      8: v := RdI32(B, off + 4 * k, Swap);
      768: v := LongWord(RdI32(B, off + 4 * k, Swap));
      16: v := RdF32(B, off + 4 * k, Swap);
      64: v := RdF64(B, off + 8 * k, Swap);
      1024: v := RdI64(B, off + 8 * k, Swap);
      1280: v := QWord(RdI64(B, off + 8 * k, Swap));
    else
      v := 0;
    end;
    AVol.Data[k] := v * slope + inter;
  end;
  Result := True;
end;

{ -------------------------------------------------------------- JNIfTI --- }

{ Numbers out of a JSON node: a flat list, a list of lists (row-major), or a
  JData-annotated array. }
function JsonNumbers(AData: TJSONData; out AVals: TI2MDoubles): Boolean;
var
  i, j: Integer;
  A: TMcxArray;
  n: Int64;
begin
  Result := False;
  SetLength(AVals, 0);
  if AData = nil then Exit;
  if AData is TJSONArray then
  begin
    for i := 0 to TJSONArray(AData).Count - 1 do
      if TJSONArray(AData).Items[i] is TJSONArray then
      begin
        for j := 0 to TJSONArray(TJSONArray(AData).Items[i]).Count - 1 do
        begin
          SetLength(AVals, Length(AVals) + 1);
          AVals[High(AVals)] := TJSONArray(TJSONArray(AData).Items[i]).Items[j].AsFloat;
        end;
      end
      else
      begin
        SetLength(AVals, Length(AVals) + 1);
        AVals[High(AVals)] := TJSONArray(AData).Items[i].AsFloat;
      end;
    Exit(Length(AVals) > 0);
  end;
  if (AData is TJSONObject) and McxDecodeJData(TJSONObject(AData), A) then
  begin
    n := McxArrayCount(A);
    SetLength(AVals, n);
    for i := 0 to n - 1 do AVals[i] := McxArrayValue(A, i);
    Exit(n > 0);
  end;
  if AData is TJSONNumber then
  begin
    SetLength(AVals, 1);
    AVals[0] := AData.AsFloat;
    Exit(True);
  end;
end;

function LoadJNifti(const AFileName: string; out AVol: TI2MVolume;
  out AError: string): Boolean;
var
  Root, Hdr, Nd, Ord_: TJSONData;
  BJ: TMcxBJData;
  S: TFileStream;
  Txt: string;
  A: TMcxArray;
  Vals: TI2MDoubles;
  Got, RowMajor: Boolean;
  i, j, k, c: Integer;
  d: array[0..3] of Integer;
  nv, src, dst: Int64;
begin
  Result := False;
  AError := '';
  Root := nil;
  BJ := nil;
  try
    if LowerCase(ExtractFileExt(AFileName)) = '.jnii' then
    begin
      S := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyWrite);
      try
        SetLength(Txt, S.Size);
        if S.Size > 0 then S.ReadBuffer(Txt[1], S.Size);
      finally
        S.Free;
      end;
      Root := GetJSON(Txt);
      Nd := nil;
      if Root is TJSONObject then Nd := TJSONObject(Root).Find('NIFTIData');
      Got := (Nd is TJSONObject) and McxDecodeJData(TJSONObject(Nd), A);
    end
    else
    begin
      BJ := TMcxBJData.Create;
      if not BJ.LoadFromFile(AFileName) then
      begin
        AError := 'cannot read ' + AFileName + ': ' + BJ.Error;
        Exit;
      end;
      Root := BJ.Root;
      Nd := nil;
      if Root is TJSONObject then Nd := TJSONObject(Root).Find('NIFTIData');
      Got := BJ.GetArray('NIFTIData', A);
    end;
    if not Got then
    begin
      AError := 'no NIFTIData array in ' + ExtractFileName(AFileName);
      Exit;
    end;
    if (Length(A.Dims) < 3) then
    begin
      AError := 'NIFTIData is not 3-D or 4-D';
      Exit;
    end;
    for i := 0 to 3 do d[i] := 1;
    for i := 0 to Min(3, High(A.Dims)) do d[i] := A.Dims[i];
    AVol.Nx := d[0];
    AVol.Ny := d[1];
    AVol.Nz := d[2];
    AVol.Nc := d[3];
    AVol.IsInteger := A.Kind in [akUInt8, akInt8, akUInt16, akInt16, akUInt32, akInt32, akUInt64, akInt64];

    { JData arrays are row-major (the first index slowest) unless _ArrayOrder_
      says column-major; the texture wants x fastest, channel-major. }
    RowMajor := True;
    Ord_ := nil;
    if Nd is TJSONObject then Ord_ := TJSONObject(Nd).Find('_ArrayOrder_');
    if (Ord_ <> nil) and (Ord_.JSONType = jtString) and
       (LowerCase(Copy(Ord_.AsString, 1, 1)) <> 'r') and (LowerCase(Copy(Ord_.AsString, 1, 1)) <> 'c') then
      RowMajor := False;
    nv := Int64(d[0]) * d[1] * d[2];
    SetLength(AVol.Data, nv * d[3]);
    for i := 0 to d[0] - 1 do
      for j := 0 to d[1] - 1 do
        for k := 0 to d[2] - 1 do
          for c := 0 to d[3] - 1 do
          begin
            if RowMajor then src := ((Int64(i) * d[1] + j) * d[2] + k) * d[3] + c
            else src := i + Int64(d[0]) * (j + Int64(d[1]) * (k + Int64(d[2]) * c));
            dst := Int64(c) * nv + i + Int64(d[0]) * (j + Int64(d[1]) * k);
            AVol.Data[dst] := McxArrayValue(A, src);
          end;

    { the affine: NIFTIHeader.Affine (3x4 or 4x4, row-major), else VoxelSize }
    for i := 0 to 2 do AVol.VoxelSize[i] := 1;
    Hdr := nil;
    if Root is TJSONObject then Hdr := TJSONObject(Root).Find('NIFTIHeader');
    if (Hdr is TJSONObject) and JsonNumbers(TJSONObject(Hdr).Find('VoxelSize'), Vals) then
      for i := 0 to Min(2, High(Vals)) do
        if Vals[i] > 0 then AVol.VoxelSize[i] := Vals[i];
    SetDiag(AVol.Affine, AVol.VoxelSize[0], AVol.VoxelSize[1], AVol.VoxelSize[2]);
    if (Hdr is TJSONObject) and JsonNumbers(TJSONObject(Hdr).Find('Affine'), Vals) and
       (Length(Vals) >= 12) then
    begin
      for i := 0 to 11 do AVol.Affine[i] := Vals[i];
      AVol.Affine[12] := 0;
      AVol.Affine[13] := 0;
      AVol.Affine[14] := 0;
      AVol.Affine[15] := 1;
    end;
    Result := True;
  finally
    if BJ <> nil then BJ.Free
    else Root.Free;
  end;
end;

function I2MLoadVolume(const AFileName: string; out AVol: TI2MVolume;
  out AError: string): Boolean;
var
  ext: string;
  k, n: Int64;
  v: Single;
begin
  AVol.Nx := 0;
  AVol.Ny := 0;
  AVol.Nz := 0;
  AVol.Nc := 0;
  SetLength(AVol.Data, 0);
  ext := LowerCase(AFileName);
  try
    if (Copy(ext, Length(ext) - 3, 4) = '.nii') or (Copy(ext, Length(ext) - 6, 7) = '.nii.gz') then
      Result := LoadNifti(AFileName, AVol, AError)
    else if (Copy(ext, Length(ext) - 4, 5) = '.jnii') or (Copy(ext, Length(ext) - 4, 5) = '.bnii') then
      Result := LoadJNifti(AFileName, AVol, AError)
    else
    begin
      AError := 'unsupported file type (want .nii, .nii.gz, .jnii, .bnii)';
      Result := False;
    end;
  except
    on E: Exception do
    begin
      AError := E.Message;
      Result := False;
    end;
  end;
  if not Result then Exit;

  n := Length(AVol.Data);
  AVol.Low := 0;
  AVol.High := 0;
  if n > 0 then
  begin
    AVol.Low := AVol.Data[0];
    AVol.High := AVol.Data[0];
  end;
  for k := 0 to n - 1 do
  begin
    v := AVol.Data[k];
    if IsNan(v) or IsInfinite(v) then Continue;
    if v < AVol.Low then AVol.Low := v;
    if v > AVol.High then AVol.High := v;
    if AVol.IsInteger then Continue;
  end;
  { a float file holding whole numbers only is a label volume too }
  if not AVol.IsInteger and (AVol.Nc = 1) and (AVol.High <= 65535) then
  begin
    AVol.IsInteger := True;
    for k := 0 to n - 1 do
      if Frac(AVol.Data[k]) <> 0 then
      begin
        AVol.IsInteger := False;
        Break;
      end;
  end;
end;

end.
