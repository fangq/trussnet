{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mdpi -- high-DPI scaling, ported from LED's Led.UI.Dpi (mima-ide, same
  author; itself from GotBox).

  On gtk2, Application.Scaled scales the forms to the Xft DPI and no further:
  it ignores the desktop's integer window-scaling factor (xfce's
  Gdk/WindowScalingFactor), and reverts a manual bump on Show. On a 4k screen
  at Xft.dpi = 150 with a factor of 2 the windows come out laid out for 150
  dpi inside a desktop drawn at 300 -- half the size of every gtk3 program.
  So on Linux v2m leaves the LCL's scaling off and scales its forms itself,
  to Xft.dpi x the factor, and hands gtk a resource style with the theme's
  font at the same factor for the text it draws on its own (menus, the
  status bar, and the size an AutoSize control is measured at as it is
  built). Windows and macOS report a true DPI and the LCL's own scaling is
  right there: everything here is then a no-op.

  V2M_SCALE, or GDK_SCALE, overrides the target as a factor of 96 dpi. }
unit i2mdpi;

{$mode objfpc}{$H+}

{$IF DEFINED(LINUX) and DEFINED(LCLGtk2)}
  {$DEFINE I2M_GTK2_CHROME}
{$ENDIF}

interface

uses
  Classes, SysUtils, Math, Controls, Forms, Graphics;

{ The PPI the forms are scaled to: the one applied, else the desktop's target. }
function I2MPPI: Integer;
{ A pixel size chosen at 96 dpi, at I2MPPI; and back. }
function I2MScale96(APixels: Integer): Integer;
function I2MScaleTo96(APixels: Integer): Integer;
{ A font AFont at APx96 pixels (chosen at 96 dpi), at the scale }
procedure I2MFontPx(AFont: TFont; APx96: Integer);
{ The theme font for the text gtk draws itself, at the scale (gtk2; once, before the
  first widget: a widget is measured as it is built). }
procedure I2MInstallChromeStyle;
{ Scale every form from the moment it is first shown (the LCL's own dialogs too).
  Once, before the first form. }
procedure I2MInstallFormScaler;
{ Scale the forms that exist to the target. Once, after the main form is made,
  before it is shown. }
procedure I2MApplyAdaptiveScale;

implementation

{$IFDEF LINUX}
uses
  Process{$IFDEF I2M_GTK2_CHROME}, Glib2, Gtk2, Pango{$ENDIF};
{$ENDIF}

var
  GFactor: Integer = -1;     { the desktop's scaling factor, read once; -1 before }
  GApplied: Integer = 0;     { the PPI the forms are at }
  GChrome: string = '';      { the font gtk was last given }
  GBaseSpec: string = '';    { the theme's own font, read before the style replaces it }
  GBaseTaken: Boolean = False;
  GBasePoints: Double = 0;   { its size in points (0: unknown) }

{ xfce's integer window-scaling factor, or 0 (gtk2 ignores it: read it as gtk3 would) }
function DesktopFactor: Integer;
{$IFDEF LINUX}
var
  Outp: string;
begin
  if GFactor >= 0 then Exit(GFactor);   { (a program run: asked for hundreds of times) }
  Result := 0;
  Outp := '';
  try
    if RunCommand('xfconf-query', ['-c', 'xsettings', '-p', '/Gdk/WindowScalingFactor'], Outp) then
      Result := StrToIntDef(Trim(Outp), 0);
  except
    Result := 0;   { no xfconf: not xfce }
  end;
  GFactor := Result;
end;
{$ELSE}
begin
  Result := 0;
end;
{$ENDIF}

function DesiredPPI: Integer;
var
  S: string;
  Fs: TFormatSettings;
  f: Double;
begin
  S := GetEnvironmentVariable('V2M_SCALE');
  if S = '' then S := GetEnvironmentVariable('GDK_SCALE');
  if S <> '' then
  begin
    Fs := DefaultFormatSettings;
    Fs.DecimalSeparator := '.';
    f := StrToFloatDef(StringReplace(S, ',', '.', []), 0, Fs);
    if f > 0 then Exit(Round(96 * f));
  end;
  Result := Screen.PixelsPerInch;   { gtk2: Xft.dpi -- what it draws fonts at }
  if Result <= 0 then Result := 96;
  if DesktopFactor >= 2 then Result := Result * DesktopFactor;
end;

function I2MPPI: Integer;
begin
  Result := GApplied;
  if Result <= 0 then Result := DesiredPPI;
end;

function I2MScale96(APixels: Integer): Integer;
begin
  Result := Round(APixels * I2MPPI / 96);
  if (APixels > 0) and (Result < 1) then Result := 1;
end;

function I2MScaleTo96(APixels: Integer): Integer;
begin
  Result := Round(APixels * 96 / I2MPPI);
end;

{ How much bigger gtk's own text must be: the target over the DPI it draws at }
function ChromeFactor: Double;
begin
  {$IFDEF I2M_GTK2_CHROME}
  if Screen.PixelsPerInch <= 0 then Exit(1.0);
  Result := I2MPPI / Screen.PixelsPerInch;
  if Result < 1.0 then Result := 1.0;
  {$ELSE}
  Result := 1.0;
  {$ENDIF}
end;

procedure I2MInstallChromeStyle;
{$IFDEF I2M_GTK2_CHROME}
var
  Style: PGtkStyle;
  P: PChar;
  D: PPangoFontDescription;
  Size: gint;
  Spec: string;
begin
  if not GBaseTaken then
  begin   { (the theme's font, before the style below makes the default style ours) }
    GBaseTaken := True;
    Style := gtk_widget_get_default_style;
    if (Style <> nil) and (Style^.font_desc <> nil) then
    begin
      GBasePoints := pango_font_description_get_size(Style^.font_desc) / PANGO_SCALE;
      if pango_font_description_get_size_is_absolute(Style^.font_desc) and (Screen.PixelsPerInch > 0) then
        GBasePoints := GBasePoints * 72 / Screen.PixelsPerInch;   { (device pixels) }
      P := pango_font_description_to_string(Style^.font_desc);
      if P <> nil then
      begin
        GBaseSpec := StrPas(P);
        g_free(P);
      end;
    end;
  end;
  if (GBaseSpec = '') or (ChromeFactor <= 1.001) then Exit;
  D := pango_font_description_from_string(PChar(GBaseSpec));
  if D = nil then Exit;
  try
    Size := pango_font_description_get_size(D);
    if Size <= 0 then Exit;
    { (points or device pixels: two different setters) }
    if pango_font_description_get_size_is_absolute(D) then
      pango_font_description_set_absolute_size(D, Size * ChromeFactor)
    else
      pango_font_description_set_size(D, Round(Size * ChromeFactor));
    P := pango_font_description_to_string(D);
    if P = nil then Exit;
    Spec := StrPas(P);
    g_free(P);
  finally
    pango_font_description_free(D);
  end;
  if Spec = GChrome then Exit;
  gtk_rc_parse_string(PChar('style "i2m_scaled"'#10'{'#10'  font_name = "' + Spec + '"'#10'}'#10 +
    'class "*" style "i2m_scaled"'#10));
  gtk_rc_reset_styles(gtk_settings_get_default);
  GChrome := Spec;
end;
{$ELSE}
begin
end;
{$ENDIF}

{ The controls with no font of their own: the style already draws them at the
  scale, and AutoAdjustLayout would read that height back and scale it again --
  so they are put back to no font after it }
procedure KeepDefaultFonts(AControl: TControl; ACollect: Boolean; AList: TFPList);
var
  i: Integer;
begin
  if AControl = nil then Exit;
  if ACollect then
  begin
    if (AControl.Font <> nil) and (AControl.Font.Height = 0) and (AControl.Font.Size = 0) then
      AList.Add(AControl);
  end
  else if AList.IndexOf(AControl) >= 0 then
    AControl.Font.Height := 0;
  if AControl is TWinControl then
    for i := 0 to TWinControl(AControl).ControlCount - 1 do
      KeepDefaultFonts(TWinControl(AControl).Controls[i], ACollect, AList);
end;

procedure I2MFontPx(AFont: TFont; APx96: Integer);
begin
  AFont.Height := -I2MScale96(APx96);
end;

{ gtk2: the fonts of AControl and its children that name a face but no size (a
  Monospace log): they fall back to an unscaled default, so they get the theme's
  pixel height at the scale. A height is AutoAdjustLayout's, already scaled; a font
  with neither is the style's (KeepDefaultFonts). }
procedure FixFonts(AControl: TControl);
var
  i: Integer;
  F: TFont;
begin
  {$IFDEF I2M_GTK2_CHROME}
  if (AControl = nil) or (Screen.PixelsPerInch <= 0) or (ChromeFactor <= 1.001) or (GBasePoints <= 0) then Exit;
  F := AControl.Font;
  if (F <> nil) and (F.Height = 0) and (F.Size = 0) and not SameText(F.Name, 'default') then
    F.Height := -Max(1, Round(GBasePoints * Screen.PixelsPerInch / 72 * ChromeFactor));
  if AControl is TWinControl then
    for i := 0 to TWinControl(AControl).ControlCount - 1 do
      FixFonts(TWinControl(AControl).Controls[i]);
  {$ENDIF}
end;

procedure ScaleFormTo(AForm: TCustomForm; ATarget: Integer);
var
  Cur: Integer;
  L: TFPList;
begin
  if (AForm = nil) or (ATarget <= 0) then Exit;
  Cur := AForm.PixelsPerInch;
  if Cur <= 0 then Cur := 96;
  if ATarget = Cur then Exit;   { (idempotent: the startup sweep and the hook cannot compound) }
  L := TFPList.Create;
  try
    KeepDefaultFonts(AForm, True, L);
    AForm.AutoAdjustLayout(lapAutoAdjustForDPI, Cur, ATarget, AForm.Width, Round(AForm.Width * ATarget / Cur));
    KeepDefaultFonts(AForm, False, L);
  finally
    L.Free;
  end;
  FixFonts(AForm);
end;

type
  TI2MFormScaler = class
    procedure FormVisibleChanged(Sender: TObject; AForm: TCustomForm);
  end;

procedure TI2MFormScaler.FormVisibleChanged(Sender: TObject; AForm: TCustomForm);
begin
  if (AForm = nil) or not AForm.Visible then Exit;
  try
    ScaleFormTo(AForm, I2MPPI);
  except
    { (cosmetic: never fatal, and maybe raised while an error dialog is shown) }
  end;
end;

var
  GScaler: TI2MFormScaler = nil;

procedure I2MInstallFormScaler;
begin
  if GScaler <> nil then Exit;
  GScaler := TI2MFormScaler.Create;
  Screen.AddHandlerFormVisibleChanged(@GScaler.FormVisibleChanged);
end;

procedure I2MApplyAdaptiveScale;
var
  i, T: Integer;
begin
  T := DesiredPPI;
  for i := 0 to Screen.CustomFormCount - 1 do
    try
      ScaleFormTo(Screen.CustomForms[i], T);
    except
    end;
  GApplied := T;
  I2MInstallChromeStyle;
end;

finalization
  FreeAndNil(GScaler);
end.
