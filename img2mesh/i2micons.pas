{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2micons -- the toolbar and window icons.

  The artwork is MCX Studio's icon set (mcx/mcxstudio2/icons, same author),
  96-pixel PNG masters in icons/, embedded as i2micons.lrs ("make icons"
  regenerates it; it is committed, so a build needs no lazres). Each is
  area-averaged down to the size the screen asks for. }
unit i2micons;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, Graphics, Forms, LResources, FPImage, FPReadPNG,
  IntfGraphics, GraphType;

{ The icon ANAme at ASize x ASize pixels, or nil. The caller owns it. }
function I2MIcon(const AName: string; ASize: Integer): TBitmap;

{ The toolbar size for this screen: 24 pixels at 96 dpi. }
function I2MIconSize: Integer;

{ The window / task-bar icon. }
procedure I2MApplyWindowIcon(const AName: string);

implementation

function LoadMaster(const AName: string): TFPMemoryImage;
var
  R: TLResource;
  S: TStringStream;
  Rd: TFPReaderPNG;
begin
  Result := nil;
  R := LazarusResources.Find(AName);
  if R = nil then Exit;
  S := TStringStream.Create(R.Value);
  Rd := TFPReaderPNG.Create;
  Result := TFPMemoryImage.Create(0, 0);
  try
    try
      Result.LoadFromStream(S, Rd);
    except
      FreeAndNil(Result);
    end;
  finally
    Rd.Free;
    S.Free;
  end;
end;

function I2MIcon(const AName: string; ASize: Integer): TBitmap;
var
  M: TFPMemoryImage;
  Img: TLazIntfImage;
  x, y, sx, sy, x0, x1, y0, y1, n: Integer;
  r, g, b, a, w: Double;
  C: TFPColor;
  Desc: TRawImageDescription;
begin
  Result := nil;
  M := LoadMaster(AName);
  if M = nil then Exit;
  Img := TLazIntfImage.Create(0, 0);
  try
    { (a local: DataDescription is a record property, and a method called
      on it would change a temporary) }
    Desc.Init_BPP32_B8G8R8A8_BIO_TTB(ASize, ASize);
    Img.DataDescription := Desc;
    for y := 0 to ASize - 1 do
      for x := 0 to ASize - 1 do
      begin
        { the master pixels under this one, colour weighted by alpha }
        x0 := x * M.Width div ASize;
        x1 := Max(x0 + 1, (x + 1) * M.Width div ASize);
        y0 := y * M.Height div ASize;
        y1 := Max(y0 + 1, (y + 1) * M.Height div ASize);
        r := 0; g := 0; b := 0; a := 0; n := 0;
        for sy := y0 to y1 - 1 do
          for sx := x0 to x1 - 1 do
          begin
            C := M.Colors[sx, sy];
            w := C.Alpha / 65535;
            r := r + C.Red * w;
            g := g + C.Green * w;
            b := b + C.Blue * w;
            a := a + C.Alpha;
            Inc(n);
          end;
        if a > 0 then
        begin
          w := a / 65535;
          C.Red := Round(r / w);
          C.Green := Round(g / w);
          C.Blue := Round(b / w);
        end
        else
        begin
          C.Red := 0; C.Green := 0; C.Blue := 0;
        end;
        C.Alpha := Round(a / n);
        Img.Colors[x, y] := C;
      end;
    Result := TBitmap.Create;
    Result.LoadFromIntfImage(Img);
  finally
    Img.Free;
    M.Free;
  end;
end;

function I2MIconSize: Integer;
begin
  Result := Max(16, Round(24 * Screen.PixelsPerInch / 96));
end;

procedure I2MApplyWindowIcon(const AName: string);
var
  B: TBitmap;
begin
  B := I2MIcon(AName, 64);
  if B = nil then Exit;
  try
    Application.Icon.Assign(B);
  finally
    B.Free;
  end;
end;

initialization
  {$I i2micons.lrs}
end.
