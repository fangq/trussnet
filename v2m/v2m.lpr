{ SPDX-License-Identifier: GPL-3.0-or-later
  v2m -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  A graphical front end for v2mesh: open an image, mesh it, look at both.

    v2m [image] [mesh] [--tn "v2mesh args"] [--run]
             [--show volume|mesh|both] [--hide L1,L2,..] [--clip xlo,xhi,ylo,yhi,zlo,zhi] [--screenshot out.png [--shot-size WxH]]

  --gl auto|glx|egl|soft picks how the view gets OpenGL (also V2M_GL):
  auto uses a GL window where the display has a GL visual and otherwise
  renders offscreen through EGL (the GPU, else Mesa's software rasteriser).

  --run meshes the image at start-up; --screenshot writes the view (after the
  run, if any) and exits, which is how it is tested without a display. }
program v2m;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  Interfaces, Forms, SysUtils, StrUtils, Classes, Process, mcxgl, i2mmain, i2micons;


type
  TI2MHeadless = class
    class procedure Report(Sender: TObject; E: Exception);
  end;

class procedure TI2MHeadless.Report(Sender: TObject; E: Exception);
begin
  WriteLn(StdErr, E.ClassName, ': ', E.Message);
  Flush(StdErr);
  Halt(2);
end;

var
  i, w, h, Page: Integer;
  a, Show, Shot, HideList, Clip, V2mArgs, Image, Mesh, Cad: string;
  RunIt: Boolean;
  f: array of string;
  Lo, Hi: TMcxVec3;
  Deadline: QWord;
  Toks: TStringList;
begin
  Shot := '';
  Show := '';
  HideList := '';
  Clip := '';
  V2mArgs := '';
  Image := '';
  Mesh := '';
  Cad := '';
  RunIt := False;
  Page := 0;
  w := 1024;
  h := 768;
  i := 1;
  while i <= ParamCount do
  begin
    a := ParamStr(i);
    if (a = '--screenshot') and (i < ParamCount) then begin Inc(i); Shot := ParamStr(i); end
    else if (a = '--shot-size') and (i < ParamCount) then
    begin
      Inc(i);
      f := ParamStr(i).Split('x');
      if Length(f) = 2 then begin w := StrToIntDef(f[0], w); h := StrToIntDef(f[1], h); end;
    end
    else if (a = '--clip') and (i < ParamCount) then begin Inc(i); Clip := ParamStr(i); end
    else if (a = '--tn') and (i < ParamCount) then begin Inc(i); V2mArgs := ParamStr(i); end
    else if (a = '--show') and (i < ParamCount) then begin Inc(i); Show := ParamStr(i); end
    else if (a = '--page') and (i < ParamCount) then begin Inc(i); Page := StrToIntDef(ParamStr(i), 0); end
    else if (a = '--gl') and (i < ParamCount) then begin Inc(i); I2MGLMode := LowerCase(ParamStr(i)); end
    else if (a = '--hide') and (i < ParamCount) then begin Inc(i); HideList := ParamStr(i); end
    else if a = '--run' then RunIt := True
    else if AnsiIndexStr(LowerCase(ExtractFileExt(a)), ['.jmsh', '.bmsh', '.off', '.stl']) >= 0 then Mesh := a
    else if AnsiIndexStr(LowerCase(ExtractFileExt(a)), ['.step', '.stp', '.poly', '.smesh']) >= 0 then Cad := a
    else Image := a;
    Inc(i);
  end;

  if (I2MGLMode = 'auto') and (GetEnvironmentVariable('V2M_GL') <> '') then
    I2MGLMode := LowerCase(GetEnvironmentVariable('V2M_GL'));
  RequireDerivedFormResource := False;
  Application.Title := 'v2m';
  Application.Scaled := True;
  if Shot <> '' then Application.OnException := @TI2MHeadless.Report;
  Application.Initialize;
  I2MApplyWindowIcon('tetmesh');
  I2MMainForm := TI2MMainForm.Create(Application);
  I2MMainForm.EchoLog := Shot <> '';
  I2MMainForm.Show;
  Application.ProcessMessages;
  if Image <> '' then I2MMainForm.LoadImage(Image);
  if Mesh <> '' then I2MMainForm.LoadMesh(Mesh);
  if Cad <> '' then I2MMainForm.LoadCad(Cad);
  if Page > 0 then I2MMainForm.ShowPage(Page);
  if HideList <> '' then I2MMainForm.HideLabels(HideList);
  if Show <> '' then I2MMainForm.ShowOnly(Show);
  if V2mArgs <> '' then
  begin   { each flag into its field (unknown ones into "Other arguments") }
    Toks := TStringList.Create;
    try
      CommandToList(V2mArgs, Toks);
      i := 0;
      while i < Toks.Count do
      begin
        if (i + 1 < Toks.Count) and ((Toks[i + 1] = '') or (Toks[i + 1][1] <> '-') or
           (StrToFloatDef(Toks[i + 1], 1e300) <> 1e300)) then
        begin
          I2MMainForm.SetOption(Toks[i], Toks[i + 1]);
          Inc(i, 2);
        end
        else
        begin
          I2MMainForm.SetOption(Toks[i], '');
          Inc(i);
        end;
      end;
    finally
      Toks.Free;
    end;
  end;
  if Clip <> '' then
  begin
    f := Clip.Split(',');
    if Length(f) = 6 then
    begin
      Lo := McxVec3(StrToFloatDef(f[0], 0), StrToFloatDef(f[2], 0), StrToFloatDef(f[4], 0));
      Hi := McxVec3(StrToFloatDef(f[1], 1), StrToFloatDef(f[3], 1), StrToFloatDef(f[5], 1));
      I2MMainForm.SetClip(Lo, Hi);
    end;
  end;
  if RunIt then
  begin
    I2MMainForm.Run;
    if Shot <> '' then
    begin
      Deadline := GetTickCount64 + 600000;
      repeat
        Application.ProcessMessages;
        Sleep(20);
      until (not I2MMainForm.Running and not I2MMainForm.Busy) or (GetTickCount64 > Deadline);
    end;
  end;
  if Shot <> '' then
  begin
    Application.ProcessMessages;
    I2MMainForm.FitView(False);   { (the picture has no cards) }
    if I2MMainForm.SaveImage(Shot, w, h) then WriteLn('wrote ', Shot)
    else begin WriteLn(StdErr, 'could not render ', Shot); Halt(1); end;
    Application.ProcessMessages;
    Halt(0);
  end;
  Application.Run;
end.
