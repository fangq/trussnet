{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2mabout -- Help > About: the title, version, license and author, what v2m
  and the v2mesh it runs include, and the works they build on.

  A designed form (i2mabout.lfm); the text is in its InfoMemo, so it can be
  edited in the Lazarus designer. Only the versions are filled in at run time:
  v2m's own (I2MVersion, kept with v2mesh's: CMakeLists.txt's project
  VERSION) and the engine's, asked of the v2mesh the Meshing panel names. }
unit i2mabout;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, StdCtrls, ExtCtrls, Process, LCLIntf, i2micons;

const
  I2MVersion = '0.5.0';
  I2MHomePage = 'https://github.com/fangq/trussnet';

type
  TI2MAboutForm = class(TForm)
    HeadPanel: TPanel;
    AppIcon: TImage;
    TitleLabel, SubtitleLabel, VersionLabel, EngineLabel: TLabel;
    InfoMemo: TMemo;
    FootPanel: TPanel;
    HomeLink: TLabel;
    OkButton: TButton;
    procedure FormCreate(Sender: TObject);
    procedure HomeLinkClick(Sender: TObject);
  end;

{ The About dialog, modal; AEngine: the v2mesh executable (its version shown). }
procedure I2MShowAbout(AOwner: TComponent; const AEngine: string);

implementation

{$R *.lfm}

procedure TI2MAboutForm.FormCreate(Sender: TObject);
var
  B: TBitmap;
begin
  B := I2MIcon('tetmesh', AppIcon.Width);
  if B <> nil then
  try
    AppIcon.Picture.Assign(B);
  finally
    B.Free;
  end;
  VersionLabel.Caption := 'Version ' + I2MVersion;
  HomeLink.Caption := I2MHomePage;
  InfoMemo.SelStart := 0;
end;

procedure TI2MAboutForm.HomeLinkClick(Sender: TObject);
begin
  OpenURL(I2MHomePage);
end;

procedure I2MShowAbout(AOwner: TComponent; const AEngine: string);
var
  F: TI2MAboutForm;
  s: string;
begin
  F := TI2MAboutForm.Create(AOwner);
  try
    s := '';
    if (AEngine <> '') and FileExists(AEngine) then
      try
        if RunCommand(AEngine, ['--version'], s, [poNoConsole]) then s := Trim(s) else s := '';
      except
        s := '';
      end;
    if s <> '' then F.EngineLabel.Caption := 'Engine: ' + s + '  (' + AEngine + ')'
    else F.EngineLabel.Caption := 'Engine: v2mesh not found -- set its path in Meshing > v2mesh path';
    F.ShowModal;
  finally
    F.Free;
  end;
end;

end.
