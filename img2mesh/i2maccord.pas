{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2maccord -- the left-hand accordion, in the style of mcxstudio2's section
  navigator (same author): each section is a large, bold, pill-shaped title
  with a chevron, and a body under it; one section is open at a time. Sections
  come in groups (meshing, display), each with its own pill colours, under a
  captioned separator.

  The title is a graphic control that paints its own pill (mcxstudio2 found a
  flat speed button's "hot" face painted over any band drawn behind it, so it
  swapped in a button that draws nothing but its caption; a control that owns
  all of its paint has no such face to lose to). The titles share one font
  size: the largest at which every caption fits on one line. }
unit i2maccord;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, LCLType, LCLIntf, Controls, ExtCtrls, Forms, Graphics, GraphType;

type
  TI2MSectionHead = class(TGraphicControl)
  private
    FExpanded, FHot: Boolean;
    FScheme: Integer;
    FFontPx: Integer;
    procedure SetExpanded(AValue: Boolean);
  protected
    procedure Paint; override;
    procedure MouseEnter; override;
    procedure MouseLeave; override;
  public
    constructor Create(AOwner: TComponent); override;
    property Expanded: Boolean read FExpanded write SetExpanded;
    property Scheme: Integer read FScheme write FScheme;   { 0 meshing, 1 display }
    property FontPx: Integer read FFontPx write FFontPx;
  published
    property Caption;
    property OnClick;
  end;

  { a group's caption over a rule }
  TI2MSeparator = class(TGraphicControl)
  protected
    procedure Paint; override;
  public
    constructor Create(AOwner: TComponent); override;
  published
    property Caption;
  end;

  TI2MAccordion = class(TScrollBox)
  private
    FHeads: array of TI2MSectionHead;
    FBodies: array of TPanel;
    FOrder, FScheme: Integer;
    procedure HeadClick(Sender: TObject);
    function NextTop: Integer;
    procedure FitTitles;
  protected
    procedure Resize; override;
  public
    constructor Create(AOwner: TComponent); override;
    { A captioned rule; the sections after it take pill colours AScheme. }
    procedure AddGroup(const ACaption: string; AScheme: Integer);
    { A new section at the bottom; returns its body, for the caller's controls
      (aligned alTop, in the order they are made). }
    function AddSection(const ACaption: string): TPanel;
    { Opens section AIndex (closing the others); -1 closes all. }
    procedure Open(AIndex: Integer);
    function IndexOf(const ACaption: string): Integer;
    function Count: Integer;
    function Opened: Integer;
  end;

implementation

function Scaled(A: Integer): Integer;
begin
  Result := MulDiv(A, Screen.PixelsPerInch, 96);
end;

const
  { per scheme: open, open under the pointer, closed, closed under the
    pointer (BGR). 0: mcxstudio2's accent blue and grey; 1: a teal }
  Fills: array[0..1, 0..3] of TColor = (
    ($00DCA679, $00E6B78E, $00D6D6D6, $00E4E4E4),
    ($009FBB66, $00B2CB80, $00DAE4D2, $00E6EEE0));
  Ink = $001E1E1E;
  TitlePx = 21;      { the title font at 96 dpi, before fitting }
  TitleMinPx = 13;

{ ------------------------------------------------------- TI2MSectionHead --- }

constructor TI2MSectionHead.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  Height := Scaled(40);
  Cursor := crHandPoint;
  FFontPx := Scaled(TitlePx);
  ControlStyle := ControlStyle + [csClickEvents];
end;

procedure TI2MSectionHead.SetExpanded(AValue: Boolean);
begin
  if FExpanded = AValue then Exit;
  FExpanded := AValue;
  Invalidate;
end;

procedure TI2MSectionHead.MouseEnter;
begin
  inherited MouseEnter;
  FHot := True;
  Invalidate;
end;

procedure TI2MSectionHead.MouseLeave;
begin
  inherited MouseLeave;
  FHot := False;
  Invalidate;
end;

procedure TI2MSectionHead.Paint;
var
  C: TCanvas;
  Pad, R, cx, cy, s, k: Integer;
begin
  C := Canvas;
  C.AntialiasingMode := amOn;
  { the margin around the pill: the parent's colour }
  C.Brush.Style := bsSolid;
  C.Brush.Color := Parent.Brush.Color;
  C.FillRect(0, 0, Width, Height);
  if FExpanded then k := 0 else k := 2;
  if FHot then Inc(k);
  Pad := Scaled(4);
  { RoundRect's last two arguments are the ellipse's width and height, not a
    radius: the full height makes each end a semicircle (a pill) }
  R := Height - 2 * Scaled(2);
  C.Brush.Color := Fills[FScheme, k];
  C.Pen.Style := psSolid;
  C.Pen.Color := Fills[FScheme, k];
  C.RoundRect(Pad, Scaled(2), Width - Pad, Height - Scaled(2), R, R);

  { the chevron: two strokes, > closed, v open }
  C.Pen.Color := Ink;
  C.Pen.Width := Max(2, Scaled(3));
  cx := Pad + Scaled(18);
  cy := Height div 2;
  s := Scaled(6);
  if FExpanded then
  begin
    C.Line(cx - s, cy - s div 2, cx, cy + s div 2);
    C.Line(cx, cy + s div 2, cx + s, cy - s div 2);
  end
  else
  begin
    C.Line(cx - s div 2, cy - s, cx + s div 2, cy);
    C.Line(cx + s div 2, cy, cx - s div 2, cy + s);
  end;
  C.Pen.Width := 1;

  C.Brush.Style := bsClear;
  C.Font.Assign(Font);
  C.Font.Style := [fsBold];
  C.Font.Height := -FFontPx;
  C.Font.Color := Ink;
  C.TextOut(cx + Scaled(18), (Height - C.TextHeight(Caption)) div 2, Caption);
end;

{ --------------------------------------------------------- TI2MSeparator --- }

constructor TI2MSeparator.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  Height := Scaled(28);
end;

procedure TI2MSeparator.Paint;
var
  C: TCanvas;
  tw, y, x0: Integer;
begin
  C := Canvas;
  C.Brush.Style := bsSolid;
  C.Brush.Color := Parent.Brush.Color;
  C.FillRect(0, 0, Width, Height);
  C.Font.Assign(Font);
  C.Font.Style := [fsBold];
  C.Font.Height := -Scaled(13);
  C.Font.Color := $00707070;
  x0 := Scaled(10);
  tw := C.TextWidth(UpperCase(Caption));
  y := Height - Scaled(8);
  C.Brush.Style := bsClear;
  C.TextOut(x0, y - C.TextHeight('X') div 2 - 1, UpperCase(Caption));
  C.Pen.Color := $00A0A0A0;
  C.Pen.Width := 1;
  C.Line(x0 + tw + Scaled(8), y, Width - Scaled(8), y);
end;

{ --------------------------------------------------------- TI2MAccordion --- }

constructor TI2MAccordion.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  BorderStyle := bsNone;
  HorzScrollBar.Visible := False;
  VertScrollBar.Tracking := True;
  VertScrollBar.Increment := Scaled(24);
end;

function TI2MAccordion.NextTop: Integer;
begin
  Inc(FOrder, 10);
  Result := FOrder;
end;

procedure TI2MAccordion.AddGroup(const ACaption: string; AScheme: Integer);
var
  G: TI2MSeparator;
begin
  G := TI2MSeparator.Create(Self);
  G.Parent := Self;
  G.Align := alTop;
  G.Top := NextTop;
  G.Caption := ACaption;
  FScheme := AScheme;
end;

function TI2MAccordion.AddSection(const ACaption: string): TPanel;
var
  H: TI2MSectionHead;
  k: Integer;
begin
  H := TI2MSectionHead.Create(Self);
  H.Parent := Self;
  H.Align := alTop;
  H.Top := NextTop;
  H.Caption := ACaption;
  H.Scheme := FScheme;
  H.OnClick := @HeadClick;
  Result := TPanel.Create(Self);
  Result.Parent := Self;
  Result.Align := alTop;
  Result.Top := NextTop;
  Result.BevelOuter := bvNone;
  Result.AutoSize := True;
  Result.BorderSpacing.Left := Scaled(8);
  Result.BorderSpacing.Right := Scaled(4);
  Result.BorderSpacing.Bottom := Scaled(6);
  Result.Visible := False;
  k := Length(FHeads);
  SetLength(FHeads, k + 1);
  SetLength(FBodies, k + 1);
  FHeads[k] := H;
  FBodies[k] := Result;
  H.Tag := k;
  FitTitles;
end;

{ one title size for all: the largest (up to TitlePx) at which the widest
  caption still fits its pill }
procedure TI2MAccordion.FitTitles;
var
  B: TBitmap;
  px, k, avail: Integer;
  Fits: Boolean;
begin
  if Length(FHeads) = 0 then Exit;
  avail := ClientWidth - Scaled(4 + 18 + 18 + 10);   { pad, chevron, gap, the pill's round end }
  if avail < Scaled(60) then Exit;
  B := TBitmap.Create;
  try
    B.SetSize(4, 4);
    B.Canvas.Font.Assign(Font);
    B.Canvas.Font.Style := [fsBold];
    px := Scaled(TitlePx);
    repeat
      B.Canvas.Font.Height := -px;
      Fits := True;
      for k := 0 to High(FHeads) do
        Fits := Fits and (B.Canvas.TextWidth(FHeads[k].Caption) <= avail);
      if not Fits then Dec(px);
    until Fits or (px <= Scaled(TitleMinPx));
  finally
    B.Free;
  end;
  for k := 0 to High(FHeads) do
    if FHeads[k].FontPx <> px then
    begin
      FHeads[k].FontPx := px;
      FHeads[k].Invalidate;
    end;
end;

procedure TI2MAccordion.Resize;
begin
  inherited Resize;
  FitTitles;
end;

procedure TI2MAccordion.HeadClick(Sender: TObject);
var
  k: Integer;
begin
  k := TComponent(Sender).Tag;
  if FHeads[k].Expanded then Open(-1) else Open(k);
end;

procedure TI2MAccordion.Open(AIndex: Integer);
var
  k: Integer;
begin
  DisableAlign;
  try
    for k := 0 to High(FHeads) do
    begin
      FHeads[k].Expanded := k = AIndex;
      { a hidden alTop panel keeps the Top it had, and is put back by it: just
        past its own title's Top (at the title's bottom it would tie with the
        next title, and the tie went the wrong way), or it lands wherever its
        old Top now falls }
      if k = AIndex then FBodies[k].Top := FHeads[k].Top + 1;
      FBodies[k].Visible := k = AIndex;
    end;
  finally
    EnableAlign;
  end;
  if (AIndex >= 0) and (AIndex <= High(FHeads)) then ScrollInView(FHeads[AIndex])
  else VertScrollBar.Position := 0;
end;

function TI2MAccordion.IndexOf(const ACaption: string): Integer;
var
  k: Integer;
begin
  for k := 0 to High(FHeads) do
    if SameText(FHeads[k].Caption, ACaption) then Exit(k);
  Result := -1;
end;

function TI2MAccordion.Count: Integer;
begin
  Result := Length(FHeads);
end;

function TI2MAccordion.Opened: Integer;
var
  k: Integer;
begin
  for k := 0 to High(FHeads) do
    if FHeads[k].Expanded then Exit(k);
  Result := -1;
end;

end.
