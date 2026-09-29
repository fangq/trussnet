(* SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mshapes -- JSON shape constructs (MCX Shapes; JMesh ShapeXxx / CSGXxx): a
  document to edit (the Shapes panel), and its preview -- each construct as
  translucent triangles coloured by its Tag, as MCX Studio draws them.

  The document keeps its objects in order (later ones overwrite earlier ones)
  and the style it came in: {"Shapes": [{Key: body}, ..]} (MCX) or an object
  of keys (JMesh). The drawing follows v2mesh's reading of each construct
  (src/v2m_sdfshape.cpp): MCX Box / Subgrid (1-based) / Layers (1-based,
  inclusive) in voxels, Grid the domain [0, Size]; JMesh corners O / P.
  Slabs, layers and planes are cut to the domain (the Grid, else object 1's
  bounds). A CSG object draws its operands; what a CSGSubtract takes away
  is drawn faint. The drawing is a preview: v2mesh's result is exact. *)
unit i2mshapes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpjson, jsonparser, i2mmesh;

type
  TI2MShapeTri = record
    A, B, C: TI2MPoint;
    Tag: Integer;
    Alpha: Single;
  end;

  TI2MShapeScene = record
    Tris: array of TI2MShapeTri;
    Lo, Hi: TI2MPoint;   { the domain }
    Error: string;       { a construct that could not be drawn (the others are) }
  end;

  TI2MShapeDoc = class
  private
    FItems: TJSONArray;    (* [{Key: body}, ..], in order *)
    FExtra: TJSONObject;   { the root's other members (Clip, Session, ..) }
    FArrayStyle: Boolean;  (* saved as {"Shapes": [..]} (MCX), else as an object of keys (JMesh) *)
  public
    constructor Create;
    destructor Destroy; override;
    { a new document: MCX Studio's, a 60-voxel Grid of label 1 (AGrid false: empty) }
    procedure Clear(AGrid: Boolean = True);
    function LoadFromText(const AText: string; out AErr: string): Boolean;
    function LoadFromFile(const AFileName: string; out AErr: string): Boolean;
    function AsText: string;
    procedure SaveToFile(const AFileName: string);
    function Count: Integer;
    function Key(AIndex: Integer): string;
    function Body(AIndex: Integer): TJSONData;
    { ABody is taken over; a key already there (JMesh style) gets a "(n)" name }
    procedure Add(const AKey: string; ABody: TJSONData);
    procedure Delete(AIndex: Integer);
    procedure Move(AFrom, ATo: Integer);
    function MaxTag: Integer;
    { the preview; ASelected (-1: none) is drawn more opaque }
    function Build(ASelected: Integer): TI2MShapeScene;
  end;

{ a new construct's body (JSON text), placed in the domain ALo .. AHi }
function I2MDefaultShape(const AKey: string; const ALo, AHi: TI2MPoint; ATag: Integer): string;

const
  I2MMCXShapes: array[0..10] of string = ('Grid', 'Box', 'Subgrid', 'Sphere', 'Cylinder',
    'XSlabs', 'YSlabs', 'ZSlabs', 'XLayers', 'YLayers', 'ZLayers');
  I2MJMeshShapes: array[0..9] of string = ('ShapeBox3', 'ShapeSphere', 'ShapeCylinder', 'ShapeCone',
    'ShapeConeFrustum', 'ShapeEllipsoid', 'ShapeTorus', 'ShapeSphereShell', 'ShapeSphereSegment', 'ShapePlane3');

implementation

const
  Normal = 0.45;   { a construct's opacity }
  Chosen = 0.8;    { the selected one's }
  Faint = 0.12;    { what a CSGSubtract takes away }

function P3(x, y, z: Double): TI2MPoint;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

{ the key without its "(name)", and the name }
function BaseKey(const AKey: string; out AName: string): string;
var
  p: Integer;
begin
  p := Pos('(', AKey);
  if (p > 0) and (AKey[Length(AKey)] = ')') then
  begin
    AName := Copy(AKey, p + 1, Length(AKey) - p - 1);
    Result := Copy(AKey, 1, p - 1);
  end
  else
  begin
    AName := '';
    Result := AKey;
  end;
end;

{ ------------------------------------------------------------ the document --- }

constructor TI2MShapeDoc.Create;
begin
  FItems := TJSONArray.Create;
  FExtra := TJSONObject.Create;
  Clear;
end;

destructor TI2MShapeDoc.Destroy;
begin
  FItems.Free;
  FExtra.Free;
  inherited Destroy;
end;

procedure TI2MShapeDoc.Clear(AGrid: Boolean);
begin
  FItems.Clear;
  FExtra.Clear;
  FArrayStyle := True;
  if AGrid then
    FItems.Add(TJSONObject.Create(['Grid', TJSONObject.Create(['Tag', 1, 'Size', TJSONArray.Create([60, 60, 60])])]));
end;

function TI2MShapeDoc.LoadFromText(const AText: string; out AErr: string): Boolean;
var
  Root, List: TJSONData;
  i, j: Integer;
  O: TJSONObject;
begin
  Result := False;
  AErr := '';
  try
    Root := GetJSON(AText);
  except
    on E: Exception do
    begin
      AErr := 'not valid JSON: ' + E.Message;
      Exit;
    end;
  end;
  try
    FItems.Clear;
    FExtra.Clear;
    List := Root;
    if (Root is TJSONObject) and (TJSONObject(Root).IndexOfName('Shapes') >= 0) then
    begin
      List := TJSONObject(Root).Elements['Shapes'];
      for i := 0 to TJSONObject(Root).Count - 1 do
        if TJSONObject(Root).Names[i] <> 'Shapes' then
          FExtra.Add(TJSONObject(Root).Names[i], TJSONObject(Root).Items[i].Clone);
    end;
    FArrayStyle := List is TJSONArray;
    if List is TJSONArray then
    begin
      for i := 0 to List.Count - 1 do
        if List.Items[i] is TJSONObject then
        begin
          O := TJSONObject(List.Items[i]);
          for j := 0 to O.Count - 1 do
            FItems.Add(TJSONObject.Create([O.Names[j], O.Items[j].Clone]));
        end;
    end
    else if List is TJSONObject then
    begin
      O := TJSONObject(List);
      for j := 0 to O.Count - 1 do
        if O.Names[j] = 'Clip' then FExtra.Add('Clip', O.Items[j].Clone)
        else FItems.Add(TJSONObject.Create([O.Names[j], O.Items[j].Clone]));
    end
    else
    begin
      AErr := 'the shapes must be an array or an object';
      Exit;
    end;
    Result := True;
  finally
    Root.Free;
  end;
end;

function TI2MShapeDoc.LoadFromFile(const AFileName: string; out AErr: string): Boolean;
var
  S: TStringList;
begin
  S := TStringList.Create;
  try
    try
      S.LoadFromFile(AFileName);
    except
      on E: Exception do
      begin
        AErr := E.Message;
        Exit(False);
      end;
    end;
    Result := LoadFromText(S.Text, AErr);
  finally
    S.Free;
  end;
end;

function TI2MShapeDoc.AsText: string;
var
  Root: TJSONObject;
  A: TJSONArray;
  i: Integer;
begin
  Root := TJSONObject(FExtra.Clone);
  try
    if FArrayStyle then
    begin
      A := TJSONArray.Create;
      for i := 0 to FItems.Count - 1 do A.Add(FItems.Items[i].Clone);
      Root.Add('Shapes', A);
    end
    else
      for i := 0 to FItems.Count - 1 do Root.Add(Key(i), Body(i).Clone);
    Result := Root.FormatJSON([], 2);
  finally
    Root.Free;
  end;
end;

procedure TI2MShapeDoc.SaveToFile(const AFileName: string);
var
  S: TStringList;
begin
  S := TStringList.Create;
  try
    S.Text := AsText;
    S.SaveToFile(AFileName);
  finally
    S.Free;
  end;
end;

function TI2MShapeDoc.Count: Integer;
begin
  Result := FItems.Count;
end;

function TI2MShapeDoc.Key(AIndex: Integer): string;
begin
  Result := TJSONObject(FItems.Items[AIndex]).Names[0];
end;

function TI2MShapeDoc.Body(AIndex: Integer): TJSONData;
begin
  Result := TJSONObject(FItems.Items[AIndex]).Items[0];
end;

procedure TI2MShapeDoc.Add(const AKey: string; ABody: TJSONData);
var
  k, n: Integer;
  s: string;

  function Taken(const AName: string): Boolean;
  var
    i: Integer;
  begin
    for i := 0 to Count - 1 do
      if Key(i) = AName then Exit(True);
    Result := False;
  end;

begin
  s := AKey;
  if not FArrayStyle then   { (an object's keys must differ) }
  begin
    n := 1;
    while Taken(s) do
    begin
      Inc(n);
      s := Format('%s(%d)', [AKey, n]);
    end;
  end;
  k := FItems.Add(TJSONObject.Create([s, ABody]));
  if k < 0 then ABody.Free;
end;

procedure TI2MShapeDoc.Delete(AIndex: Integer);
begin
  if (AIndex >= 0) and (AIndex < Count) then FItems.Delete(AIndex);
end;

procedure TI2MShapeDoc.Move(AFrom, ATo: Integer);
begin
  if (AFrom < 0) or (AFrom >= Count) or (ATo < 0) or (ATo >= Count) or (AFrom = ATo) then Exit;
  FItems.Move(AFrom, ATo);
end;

function TI2MShapeDoc.MaxTag: Integer;

  { the Tags in a construct (a CSG's too) }
  procedure Scan(D: TJSONData);
  var
    i: Integer;
  begin
    if D is TJSONObject then
    begin
      for i := 0 to D.Count - 1 do
        if (TJSONObject(D).Names[i] = 'Tag') and (D.Items[i] is TJSONNumber) then
          Result := Max(Result, D.Items[i].AsInteger)
        else Scan(D.Items[i]);
    end
    else if D is TJSONArray then
      for i := 0 to D.Count - 1 do Scan(D.Items[i]);
  end;

var
  i, j: Integer;
  nm: string;
  B: TJSONData;
begin
  Result := 0;
  for i := 0 to Count - 1 do
  begin
    B := Body(i);
    if (Pos('Layers', BaseKey(Key(i), nm)) = 2) and (B is TJSONArray) then
    begin   { Layers: rows [lo, hi, tag] }
      for j := 0 to B.Count - 1 do
        if (B.Items[j] is TJSONArray) and (B.Items[j].Count >= 3) and (B.Items[j].Items[2] is TJSONNumber) then
          Result := Max(Result, B.Items[j].Items[2].AsInteger);
    end
    else Scan(B);
  end;
end;

{ -------------------------------------------------------------- the preview --- }

type
  TVec = array[0..2] of Double;

  { the tessellation, in one place }
  TBuilder = class
    S: TI2MShapeScene;
    Doc: TI2MShapeDoc;
    Named: TStringList;       { name -> item index }
    Referenced: TStringList;  { the names a CSG uses: drawn there, not on their own }
    DomLo, DomHi: TVec;       { the domain (the Grid, else object 1's bounds) }
    Tag: Integer;
    Alpha: Single;
    Segs: Integer;
    procedure Tri(const A, B, C: TVec);
    procedure Quad(const A, B, C, D: TVec);
    procedure Box(const Lo, Hi: TVec);
    procedure Ellipsoid(const C: TVec; rx, ry, rz: Double; const M: array of Double);
    procedure Frustum(const A, B: TVec; ra, rb: Double);
    procedure Torus(const C, N: TVec; R, rt: Double);
    procedure Segment(const C: TVec; R: Double; const N: TVec; h1, h2: Double);
    procedure PlaneCut(const O, N: TVec);
    procedure Construct(const AKey: string; ABody: TJSONData);
    procedure Operand(D: TJSONData);
    procedure FindReferences(D: TJSONData);
  end;

function V(x, y, z: Double): TVec;
begin
  Result[0] := x;
  Result[1] := y;
  Result[2] := z;
end;

function VAdd(const A, B: TVec; s: Double = 1): TVec;
begin
  Result := V(A[0] + s * B[0], A[1] + s * B[1], A[2] + s * B[2]);
end;

function VCross(const A, B: TVec): TVec;
begin
  Result := V(A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]);
end;

function VNorm(const A: TVec): TVec;
var
  l: Double;
begin
  l := Sqrt(Sqr(A[0]) + Sqr(A[1]) + Sqr(A[2]));
  if l < 1e-30 then Exit(V(0, 0, 1));
  Result := V(A[0] / l, A[1] / l, A[2] / l);
end;

{ two unit vectors across N (N, U, W a right-handed frame) }
procedure Across(const N: TVec; out U, W: TVec);
begin
  if Abs(N[0]) < 0.9 then U := VNorm(VCross(N, V(1, 0, 0)))
  else U := VNorm(VCross(N, V(0, 1, 0)));
  W := VCross(N, U);
end;

function Num(D: TJSONData; const AName: string; ADefault: Double): Double;
var
  E: TJSONData;
begin
  Result := ADefault;
  if not (D is TJSONObject) then Exit;
  E := TJSONObject(D).Find(AName);
  if E is TJSONNumber then Result := E.AsFloat
  else if (E is TJSONArray) and (E.Count > 0) and (E.Items[0] is TJSONNumber) then Result := E.Items[0].AsFloat;
end;

function Vec3(D: TJSONData; const AName: string; out AV: TVec): Boolean;
var
  E: TJSONData;
  i: Integer;
begin
  Result := False;
  AV := V(0, 0, 0);
  if not (D is TJSONObject) then Exit;
  E := TJSONObject(D).Find(AName);
  if not (E is TJSONArray) or (E.Count < 3) then Exit;
  for i := 0 to 2 do
  begin
    if not (E.Items[i] is TJSONNumber) then Exit;
    AV[i] := E.Items[i].AsFloat;
  end;
  Result := True;
end;

function Pair(D: TJSONData; const AName: string; out a, b: Double): Boolean;
var
  E: TJSONData;
begin
  Result := False;
  a := 0;
  b := 0;
  if not (D is TJSONObject) then Exit;
  E := TJSONObject(D).Find(AName);
  if not (E is TJSONArray) or (E.Count < 2) or not (E.Items[0] is TJSONNumber) or not (E.Items[1] is TJSONNumber) then Exit;
  a := E.Items[0].AsFloat;
  b := E.Items[1].AsFloat;
  Result := True;
end;

procedure TBuilder.Tri(const A, B, C: TVec);
var
  k: Integer;
begin
  k := Length(S.Tris);
  SetLength(S.Tris, k + 1);
  S.Tris[k].A := P3(A[0], A[1], A[2]);
  S.Tris[k].B := P3(B[0], B[1], B[2]);
  S.Tris[k].C := P3(C[0], C[1], C[2]);
  S.Tris[k].Tag := Tag;
  S.Tris[k].Alpha := Alpha;
end;

procedure TBuilder.Quad(const A, B, C, D: TVec);
begin
  Tri(A, B, C);
  Tri(A, C, D);
end;

procedure TBuilder.Box(const Lo, Hi: TVec);
var
  P: array[0..7] of TVec;
  c: Integer;
begin
  for c := 0 to 7 do
    P[c] := V(IfThen(c and 1 <> 0, Hi[0], Lo[0]), IfThen(c and 2 <> 0, Hi[1], Lo[1]), IfThen(c and 4 <> 0, Hi[2], Lo[2]));
  Quad(P[0], P[2], P[3], P[1]);   { z lo }
  Quad(P[4], P[5], P[7], P[6]);   { z hi }
  Quad(P[0], P[1], P[5], P[4]);   { y lo }
  Quad(P[2], P[6], P[7], P[3]);   { y hi }
  Quad(P[0], P[4], P[6], P[2]);   { x lo }
  Quad(P[1], P[3], P[7], P[5]);   { x hi }
end;

{ a unit sphere scaled by rx, ry, rz, turned by M (3x3, row-major, local ->
  world), about C }
procedure TBuilder.Ellipsoid(const C: TVec; rx, ry, rz: Double; const M: array of Double);
var
  i, j, nu, nv: Integer;
  G: array of array of TVec;
  th, ph, x, y, z: Double;
begin
  nu := Segs;
  nv := Segs div 2;
  SetLength(G, nv + 1, nu + 1);
  for j := 0 to nv do
  begin
    th := Pi * j / nv;
    for i := 0 to nu do
    begin
      ph := 2 * Pi * i / nu;
      x := rx * Sin(th) * Cos(ph);
      y := ry * Sin(th) * Sin(ph);
      z := rz * Cos(th);
      G[j][i] := V(C[0] + M[0] * x + M[1] * y + M[2] * z, C[1] + M[3] * x + M[4] * y + M[5] * z,
        C[2] + M[6] * x + M[7] * y + M[8] * z);
    end;
  end;
  for j := 0 to nv - 1 do
    for i := 0 to nu - 1 do
      Quad(G[j][i], G[j + 1][i], G[j + 1][i + 1], G[j][i + 1]);
end;

{ a capped frustum from A (radius ra) to B (rb; 0: a cone's tip) }
procedure TBuilder.Frustum(const A, B: TVec; ra, rb: Double);
var
  N, U, W, Pa, Pb, Qa, Qb: TVec;
  i: Integer;
  t0, t1: Double;
begin
  N := VNorm(VAdd(B, A, -1));
  Across(N, U, W);
  for i := 0 to Segs - 1 do
  begin
    t0 := 2 * Pi * i / Segs;
    t1 := 2 * Pi * (i + 1) / Segs;
    Pa := VAdd(VAdd(A, U, ra * Cos(t0)), W, ra * Sin(t0));
    Qa := VAdd(VAdd(A, U, ra * Cos(t1)), W, ra * Sin(t1));
    Pb := VAdd(VAdd(B, U, rb * Cos(t0)), W, rb * Sin(t0));
    Qb := VAdd(VAdd(B, U, rb * Cos(t1)), W, rb * Sin(t1));
    Quad(Pa, Qa, Qb, Pb);
    if ra > 0 then Tri(A, Qa, Pa);
    if rb > 0 then Tri(B, Pb, Qb);
  end;
end;

procedure TBuilder.Torus(const C, N: TVec; R, rt: Double);
var
  U, W, Nn: TVec;
  G: array of array of TVec;
  i, j, nu, nv: Integer;
  a, b: Double;
begin
  Nn := VNorm(N);
  Across(Nn, U, W);
  nu := Segs;
  nv := Max(8, Segs div 2);
  SetLength(G, nu + 1, nv + 1);
  for i := 0 to nu do
  begin
    a := 2 * Pi * i / nu;
    for j := 0 to nv do
    begin
      b := 2 * Pi * j / nv;
      G[i][j] := VAdd(VAdd(VAdd(C, U, (R + rt * Cos(b)) * Cos(a)), W, (R + rt * Cos(b)) * Sin(a)), Nn, rt * Sin(b));
    end;
  end;
  for i := 0 to nu - 1 do
    for j := 0 to nv - 1 do
      Quad(G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1]);
end;

{ the sphere (C, R) between the heights h1 < h2 along N, with its flat caps }
procedure TBuilder.Segment(const C: TVec; R: Double; const N: TVec; h1, h2: Double);
var
  U, W, Nn: TVec;
  G: array of array of TVec;
  i, j, nu, nv: Integer;
  h, rr, a, t: Double;
begin
  if h1 > h2 then
  begin
    t := h1;
    h1 := h2;
    h2 := t;
  end;
  h1 := Max(h1, -R);
  h2 := Min(h2, R);
  if h2 <= h1 then Exit;
  Nn := VNorm(N);
  Across(Nn, U, W);
  nu := Segs;
  nv := Max(4, Segs div 2);
  SetLength(G, nv + 1, nu + 1);
  for j := 0 to nv do
  begin
    h := h1 + (h2 - h1) * j / nv;
    rr := Sqrt(Max(0, R * R - h * h));
    for i := 0 to nu do
    begin
      a := 2 * Pi * i / nu;
      G[j][i] := VAdd(VAdd(VAdd(C, Nn, h), U, rr * Cos(a)), W, rr * Sin(a));
    end;
  end;
  for j := 0 to nv - 1 do
    for i := 0 to nu - 1 do
      Quad(G[j][i], G[j][i + 1], G[j + 1][i + 1], G[j + 1][i]);
  for i := 0 to nu - 1 do   { the caps }
  begin
    Tri(VAdd(C, Nn, h1), G[0][i + 1], G[0][i]);
    Tri(VAdd(C, Nn, h2), G[nv][i], G[nv][i + 1]);
  end;
end;

{ the plane (O, N) where it crosses the domain: a convex polygon }
procedure TBuilder.PlaneCut(const O, N: TVec);
var
  P: array of TVec;
  Ang: array of Double;
  Nn, U, W, Cen, A, B: TVec;
  c, e, k, i, j, ax: Integer;
  da, db, t, sw: Double;
begin
  P := nil;
  Nn := VNorm(N);
  { the box's 12 edges against the plane }
  for ax := 0 to 2 do
    for c := 0 to 3 do
    begin
      A := DomLo;
      B := DomLo;
      A[ax] := DomLo[ax];
      B[ax] := DomHi[ax];
      if c and 1 <> 0 then
      begin
        A[(ax + 1) mod 3] := DomHi[(ax + 1) mod 3];
        B[(ax + 1) mod 3] := DomHi[(ax + 1) mod 3];
      end;
      if c and 2 <> 0 then
      begin
        A[(ax + 2) mod 3] := DomHi[(ax + 2) mod 3];
        B[(ax + 2) mod 3] := DomHi[(ax + 2) mod 3];
      end;
      da := (A[0] - O[0]) * Nn[0] + (A[1] - O[1]) * Nn[1] + (A[2] - O[2]) * Nn[2];
      db := (B[0] - O[0]) * Nn[0] + (B[1] - O[1]) * Nn[1] + (B[2] - O[2]) * Nn[2];
      if (da * db <= 0) and (da <> db) then
      begin
        t := da / (da - db);
        k := Length(P);
        SetLength(P, k + 1);
        P[k] := VAdd(A, VAdd(B, A, -1), t);
      end;
    end;
  if Length(P) < 3 then Exit;
  Cen := V(0, 0, 0);
  for e := 0 to High(P) do Cen := VAdd(Cen, P[e], 1 / Length(P));
  Across(Nn, U, W);
  SetLength(Ang, Length(P));
  for e := 0 to High(P) do
    Ang[e] := ArcTan2((P[e][0] - Cen[0]) * W[0] + (P[e][1] - Cen[1]) * W[1] + (P[e][2] - Cen[2]) * W[2],
      (P[e][0] - Cen[0]) * U[0] + (P[e][1] - Cen[1]) * U[1] + (P[e][2] - Cen[2]) * U[2]);
  for i := 1 to High(P) do   { (a handful of points: insertion sort by angle) }
  begin
    j := i;
    while (j > 0) and (Ang[j - 1] > Ang[j]) do
    begin
      sw := Ang[j];
      Ang[j] := Ang[j - 1];
      Ang[j - 1] := sw;
      A := P[j];
      P[j] := P[j - 1];
      P[j - 1] := A;
      Dec(j);
    end;
  end;
  for e := 0 to High(P) do Tri(Cen, P[e], P[(e + 1) mod Length(P)]);
end;

{ the names a CSG body uses (a string operand, or a "Key(name)") }
procedure TBuilder.FindReferences(D: TJSONData);
var
  i: Integer;
begin
  if D is TJSONString then Referenced.Add(D.AsString)
  else if (D is TJSONArray) or (D is TJSONObject) then
    for i := 0 to D.Count - 1 do FindReferences(D.Items[i]);
end;

procedure TBuilder.Operand(D: TJSONData);
var
  k: Integer;
begin
  if D is TJSONString then
  begin   { a named construct }
    k := Named.IndexOf(D.AsString);
    if k >= 0 then
    begin
      k := PtrInt(Named.Objects[k]);
      Construct(Doc.Key(k), Doc.Body(k));
    end;
  end
  else if (D is TJSONObject) and (D.Count >= 1) then
    Construct(TJSONObject(D).Names[0], D.Items[0]);
end;

procedure TBuilder.Construct(const AKey: string; ABody: TJSONData);
var
  k, nm: string;
  A, B, N, O, Sz: TVec;
  r, r1, r2, h1, h2, th, ph, ct, st, cp, sp: Double;
  i, j, ax, SaveTag: Integer;
  SaveAlpha: Single;
  Lo, Hi: TVec;
  Row: TJSONData;
begin
  k := BaseKey(AKey, nm);
  SaveTag := Tag;
  SaveAlpha := Alpha;
  try
    if ABody is TJSONObject then Tag := Round(Num(ABody, 'Tag', Tag));
    if (k = 'CSGUnion') or (k = 'CSGIntersect') or (k = 'CSGSubtract') then
    begin
      if not (ABody is TJSONArray) then Exit;
      j := 0;
      for i := 0 to ABody.Count - 1 do
      begin
        Row := ABody.Items[i];
        if (Row is TJSONObject) and (Row.Count = 1) and (TJSONObject(Row).Names[0] = '_DataInfo_') then Continue;
        if (k = 'CSGSubtract') and (j > 0) then Alpha := Faint;   { (taken away) }
        Operand(Row);
        Alpha := SaveAlpha;
        Inc(j);
      end;
    end
    else if (k = 'CSGObject') or (k = 'MeshObject') then
    begin
      if ABody is TJSONArray then
      begin
        for i := 0 to ABody.Count - 1 do   { the Tag first: it colours the whole }
          if (ABody.Items[i] is TJSONObject) and (TJSONObject(ABody.Items[i]).IndexOfName('Tag') >= 0) then
            Tag := Round(Num(ABody.Items[i], 'Tag', Tag));
        for i := 0 to ABody.Count - 1 do
        begin
          Row := ABody.Items[i];
          if (Row is TJSONObject) and (Row.Count = 1) and ((TJSONObject(Row).Names[0] = 'Tag') or
             (TJSONObject(Row).Names[0] = '_DataInfo_')) then Continue;
          Operand(Row);
          Break;
        end;
      end
      else Operand(ABody);
    end
    else if (k = 'Grid') or (k = 'ShapeGrid3') then
      { (the domain: drawn as the frame, not as a box over everything) }
    else if k = 'Box' then
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'Size', Sz) then Box(O, VAdd(O, Sz));
    end
    else if k = 'Subgrid' then
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'Size', Sz) then
      begin
        O := VAdd(O, V(1, 1, 1), -1);   { 1-based }
        Box(O, VAdd(O, Sz));
      end;
    end
    else if k = 'ShapeBox3' then
    begin
      if Vec3(ABody, 'O', A) and Vec3(ABody, 'P', B) then
        Box(V(Min(A[0], B[0]), Min(A[1], B[1]), Min(A[2], B[2])), V(Max(A[0], B[0]), Max(A[1], B[1]), Max(A[2], B[2])));
    end
    else if (k = 'Sphere') or (k = 'ShapeSphere') then
    begin
      r := Num(ABody, 'R', 0);
      if Vec3(ABody, 'O', O) and (r > 0) then Ellipsoid(O, r, r, r, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    end
    else if k = 'Cylinder' then
    begin
      if Vec3(ABody, 'C0', A) and Vec3(ABody, 'C1', B) then Frustum(A, B, Num(ABody, 'R', 0), Num(ABody, 'R', 0));
    end
    else if k = 'ShapeCylinder' then
    begin
      if Vec3(ABody, 'O', A) and Vec3(ABody, 'P', B) then Frustum(A, B, Num(ABody, 'R', 0), Num(ABody, 'R', 0));
    end
    else if k = 'ShapeCone' then   { base O (radius R), tip P }
    begin
      if Vec3(ABody, 'O', A) and Vec3(ABody, 'P', B) then Frustum(A, B, Num(ABody, 'R', 0), 0);
    end
    else if k = 'ShapeConeFrustum' then
    begin
      if Vec3(ABody, 'O', A) and Vec3(ABody, 'P', B) and Pair(ABody, 'R', r1, r2) then Frustum(A, B, r1, r2);
    end
    else if k = 'ShapeSphereShell' then
    begin
      if Vec3(ABody, 'O', O) and Pair(ABody, 'R', r1, r2) then
      begin
        Ellipsoid(O, Max(r1, r2), Max(r1, r2), Max(r1, r2), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
        Alpha := Faint;   { (the hollow) }
        Ellipsoid(O, Min(r1, r2), Min(r1, r2), Min(r1, r2), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
      end;
    end
    else if k = 'ShapeSphereSegment' then
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'N', N) and Pair(ABody, 'Height', h1, h2) then
        Segment(O, Num(ABody, 'R', 0), N, h1, h2);
    end
    else if k = 'ShapeTorus' then
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'N', N) then Torus(O, N, Num(ABody, 'R', 0), Num(ABody, 'Rtube', 0));
    end
    else if k = 'ShapeEllipsoid' then   { R [rx, ry, rz], Angle [theta, phi]: local -> world Rz(theta) Ry(phi) }
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'R', Sz) then
      begin
        th := 0;
        ph := 0;
        Pair(ABody, 'Angle', th, ph);
        if (TJSONObject(ABody).Find('Angle') is TJSONNumber) then th := Num(ABody, 'Angle', 0);
        ct := Cos(th);
        st := Sin(th);
        cp := Cos(ph);
        sp := Sin(ph);
        Ellipsoid(O, Sz[0], Sz[1], Sz[2], [ct * cp, -st, ct * sp, st * cp, ct, st * sp, -sp, 0, cp]);
      end;
    end
    else if k = 'ShapePlane3' then   { the half-space behind N: its plane, in the domain }
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'N', N) then PlaneCut(O, N);
    end
    else if k = 'Lens' then   { (an approximation: the aperture from apex to apex) }
    begin
      if Vec3(ABody, 'O', O) and Vec3(ABody, 'Dir', N) then
      begin
        N := VNorm(N);
        r := Num(ABody, 'R', 0);
        h1 := Num(TJSONObject(ABody).Find('Back'), 'D', 0);
        h2 := Num(TJSONObject(ABody).Find('Front'), 'D', 0);
        Frustum(VAdd(O, N, -h1), VAdd(O, N, h2), r, r);
      end;
    end
    else if (k = 'XSlabs') or (k = 'YSlabs') or (k = 'ZSlabs') or (k = 'XLayers') or (k = 'YLayers') or (k = 'ZLayers') then
    begin
      ax := Ord(k[1]) - Ord('X');
      if k[2] = 'S' then
      begin   (* Slabs: {Tag, Bound: [[lo, hi], ..]} *)
        Row := nil;
        if ABody is TJSONObject then Row := TJSONObject(ABody).Find('Bound');
        if Row is TJSONArray then
        begin
          if (Row.Count = 2) and (Row.Items[0] is TJSONNumber) then
          begin   { (a single [lo, hi]) }
            Lo := DomLo;
            Hi := DomHi;
            Lo[ax] := Row.Items[0].AsFloat;
            Hi[ax] := Row.Items[1].AsFloat;
            Box(Lo, Hi);
          end
          else
            for i := 0 to Row.Count - 1 do
              if (Row.Items[i] is TJSONArray) and (Row.Items[i].Count >= 2) then
              begin
                Lo := DomLo;
                Hi := DomHi;
                Lo[ax] := Row.Items[i].Items[0].AsFloat;
                Hi[ax] := Row.Items[i].Items[1].AsFloat;
                Box(Lo, Hi);
              end;
        end;
      end
      else if ABody is TJSONArray then
      begin   { Layers: [[lo, hi, tag], ..], 1-based, inclusive }
        for i := 0 to ABody.Count - 1 do
          if (ABody.Items[i] is TJSONArray) and (ABody.Items[i].Count >= 3) then
          begin
            Lo := DomLo;
            Hi := DomHi;
            Lo[ax] := ABody.Items[i].Items[0].AsFloat - 1;
            Hi[ax] := ABody.Items[i].Items[1].AsFloat;
            Tag := ABody.Items[i].Items[2].AsInteger;
            Box(Lo, Hi);
          end;
      end;
    end;
  finally
    Tag := SaveTag;
    Alpha := SaveAlpha;
  end;
end;

{ the bounds of the triangles AFrom .. ATo - 1 (false: none) }
function TriBounds(const S: TI2MShapeScene; AFrom, ATo: Integer; out ALo, AHi: TVec): Boolean;
var
  k: Integer;

  procedure Take(const P: TI2MPoint);
  begin
    ALo := V(Min(ALo[0], P.x), Min(ALo[1], P.y), Min(ALo[2], P.z));
    AHi := V(Max(AHi[0], P.x), Max(AHi[1], P.y), Max(AHi[2], P.z));
  end;

begin
  ALo := V(1e30, 1e30, 1e30);
  AHi := V(-1e30, -1e30, -1e30);
  for k := AFrom to ATo - 1 do
    if S.Tris[k].Alpha > Faint + 0.01 then   { (not what a CSGSubtract takes away) }
    begin
      Take(S.Tris[k].A);
      Take(S.Tris[k].B);
      Take(S.Tris[k].C);
    end;
  Result := AHi[0] >= ALo[0];
end;

function TI2MShapeDoc.Build(ASelected: Integer): TI2MShapeScene;
var
  B: TBuilder;
  i, k, pass, firstObj, f0, f1: Integer;
  nm, base: string;
  Sz: TVec;
  HaveDom, Infinite, Clip: Boolean;
begin
  B := TBuilder.Create;
  B.Named := TStringList.Create;
  B.Referenced := TStringList.Create;
  try
    B.Doc := Self;
    B.Segs := 32;
    B.Tag := 1;
    B.Alpha := Normal;
    Clip := not ((FExtra.IndexOfName('Clip') >= 0) and (FExtra.Elements['Clip'] is TJSONBoolean) and
      not FExtra.Booleans['Clip']);
    for i := 0 to Count - 1 do
    begin
      base := BaseKey(Key(i), nm);
      if nm <> '' then B.Named.AddObject(nm, TObject(PtrInt(i)));
      if Copy(base, 1, 3) = 'CSG' then B.FindReferences(Body(i));
    end;
    { the domain, as v2mesh's: a Grid's [0, Size]; else object 1's bounds
      (everything else is cut to it), or all objects' ("Clip": false) }
    HaveDom := False;
    firstObj := -1;
    for i := 0 to Count - 1 do
    begin
      base := BaseKey(Key(i), nm);
      if (nm <> '') and (B.Referenced.IndexOf(nm) >= 0) then Continue;   { (a building block) }
      if (base = 'Name') or (base = 'Origin') then Continue;
      if firstObj < 0 then firstObj := i;
      if not HaveDom and ((base = 'Grid') or (base = 'ShapeGrid3')) and
         (Vec3(Body(i), 'Size', Sz) or Vec3(Body(i), 'P', Sz)) then
      begin
        B.DomLo := V(0, 0, 0);
        B.DomHi := Sz;
        HaveDom := True;
      end;
    end;
    { the finite constructs first, in order; then the slabs, layers and planes,
      cut to the domain }
    f0 := 0;
    f1 := 0;
    for pass := 0 to 1 do
    begin
      if (pass = 1) and not HaveDom then
      begin
        if Clip then HaveDom := TriBounds(B.S, f0, f1, B.DomLo, B.DomHi)
        else HaveDom := TriBounds(B.S, 0, Length(B.S.Tris), B.DomLo, B.DomHi);
        if not HaveDom then
        begin
          B.DomLo := V(0, 0, 0);
          B.DomHi := V(1, 1, 1);
          HaveDom := True;
        end;
      end;
      for i := 0 to Count - 1 do
      begin
        base := BaseKey(Key(i), nm);
        if (nm <> '') and (B.Referenced.IndexOf(nm) >= 0) then Continue;
        Infinite := (Pos('Slabs', base) = 2) or (Pos('Layers', base) = 2) or (base = 'ShapePlane3');
        if Infinite <> (pass = 1) then Continue;
        if i = ASelected then B.Alpha := Chosen else B.Alpha := Normal;
        k := Length(B.S.Tris);
        try
          B.Construct(Key(i), Body(i));
        except
          on E: Exception do
            if B.S.Error = '' then B.S.Error := Key(i) + ': ' + E.Message;
        end;
        if i = firstObj then
        begin
          f0 := k;
          f1 := Length(B.S.Tris);
        end;
      end;
    end;
    Result := B.S;
    Result.Lo := P3(B.DomLo[0], B.DomLo[1], B.DomLo[2]);
    Result.Hi := P3(B.DomHi[0], B.DomHi[1], B.DomHi[2]);
  finally
    B.Named.Free;
    B.Referenced.Free;
    B.Free;
  end;
end;

{ ------------------------------------------------------ a new construct --- }

function I2MDefaultShape(const AKey: string; const ALo, AHi: TI2MPoint; ATag: Integer): string;
var
  c: array[0..2] of Double;
  s: Double;
  fs: TFormatSettings;

  function F(x: Double): string;
  begin
    Result := FloatToStrF(RoundTo(x, -2), ffGeneral, 8, 2, fs);
  end;

  function Vc(x, y, z: Double): string;
  begin
    Result := '[' + F(x) + ',' + F(y) + ',' + F(z) + ']';
  end;

begin
  fs := DefaultFormatSettings;
  fs.DecimalSeparator := '.';
  c[0] := (ALo.x + AHi.x) / 2;
  c[1] := (ALo.y + AHi.y) / 2;
  c[2] := (ALo.z + AHi.z) / 2;
  s := Max(1, Min(AHi.x - ALo.x, Min(AHi.y - ALo.y, AHi.z - ALo.z)) / 5);
  if AKey = 'Grid' then Result := Format('{"Tag":%d,"Size":[60,60,60]}', [ATag])
  else if AKey = 'Box' then
    Result := Format('{"Tag":%d,"O":%s,"Size":%s}', [ATag, Vc(c[0] - s, c[1] - s, c[2] - s), Vc(2 * s, 2 * s, 2 * s)])
  else if AKey = 'Subgrid' then
    Result := Format('{"Tag":%d,"O":%s,"Size":%s}', [ATag, Vc(c[0] - s + 1, c[1] - s + 1, c[2] - s + 1), Vc(2 * s, 2 * s, 2 * s)])
  else if AKey = 'Sphere' then Result := Format('{"Tag":%d,"O":%s,"R":%s}', [ATag, Vc(c[0], c[1], c[2]), F(s)])
  else if AKey = 'Cylinder' then
    Result := Format('{"Tag":%d,"C0":%s,"C1":%s,"R":%s}', [ATag, Vc(c[0], c[1], ALo.z), Vc(c[0], c[1], AHi.z), F(s / 2)])
  else if (AKey = 'XSlabs') or (AKey = 'YSlabs') or (AKey = 'ZSlabs') then
    Result := Format('{"Tag":%d,"Bound":[[%s,%s]]}', [ATag, F(c[Ord(AKey[1]) - Ord('X')] - s / 2),
      F(c[Ord(AKey[1]) - Ord('X')] + s / 2)])
  else if (AKey = 'XLayers') or (AKey = 'YLayers') or (AKey = 'ZLayers') then
    Result := Format('[[1,10,%d],[11,30,%d],[31,50,%d]]', [ATag, ATag + 1, ATag + 2])
  else if AKey = 'ShapeBox3' then
    Result := Format('{"O":%s,"P":%s,"Tag":%d}', [Vc(c[0] - s, c[1] - s, c[2] - s), Vc(c[0] + s, c[1] + s, c[2] + s), ATag])
  else if AKey = 'ShapeSphere' then Result := Format('{"O":%s,"R":%s,"Tag":%d}', [Vc(c[0], c[1], c[2]), F(s), ATag])
  else if (AKey = 'ShapeCylinder') or (AKey = 'ShapeCone') then
    Result := Format('{"O":%s,"P":%s,"R":%s,"Tag":%d}', [Vc(c[0], c[1], c[2] - s), Vc(c[0], c[1], c[2] + s), F(s / 2), ATag])
  else if AKey = 'ShapeConeFrustum' then
    Result := Format('{"O":%s,"P":%s,"R":[%s,%s],"Tag":%d}', [Vc(c[0], c[1], c[2] - s), Vc(c[0], c[1], c[2] + s),
      F(s / 2), F(s / 4), ATag])
  else if AKey = 'ShapeEllipsoid' then
    Result := Format('{"O":%s,"R":%s,"Angle":[0,0],"Tag":%d}', [Vc(c[0], c[1], c[2]), Vc(s, s / 2, s / 3), ATag])
  else if AKey = 'ShapeTorus' then
    Result := Format('{"O":%s,"N":[0,0,1],"R":%s,"Rtube":%s,"Tag":%d}', [Vc(c[0], c[1], c[2]), F(s), F(s / 3), ATag])
  else if AKey = 'ShapeSphereShell' then
    Result := Format('{"O":%s,"R":[%s,%s],"Tag":%d}', [Vc(c[0], c[1], c[2]), F(s), F(s / 2), ATag])
  else if AKey = 'ShapeSphereSegment' then
    Result := Format('{"O":%s,"R":%s,"N":[0,0,1],"Height":[%s,%s],"Tag":%d}', [Vc(c[0], c[1], c[2]), F(s), F(-s / 2),
      F(s / 2), ATag])
  else if AKey = 'ShapePlane3' then Result := Format('{"O":%s,"N":[0,0,1],"Tag":%d}', [Vc(c[0], c[1], c[2]), ATag])
  else Result := Format('{"Tag":%d}', [ATag]);
end;

end.
