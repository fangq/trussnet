{ SPDX-License-Identifier: GPL-3.0-or-later
  img2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>

  i2megl -- an OpenGL 3.3 context with no window, for displays that have no
  GLX visual (X2Go / NX / VNC sessions, ssh -X without GLX, ...).

  EGL gives a context straight from a device -- the GPU's (EGL_EXT_device_*),
  or Mesa's software rasteriser (llvmpipe) -- with no X server involved. The
  view renders into a framebuffer object and copies the picture into an
  ordinary widget. libEGL is loaded at run time, so nothing here is needed to
  start img2mesh where GLX works.

  Also: I2MGlxUsable, whether the X display can give the window a GL visual at
  all (the LCL's GL control raises inside handle creation otherwise). }
unit i2megl;

{$mode objfpc}{$H+}

interface

type
  TI2MEglMode = (emAuto, emSoftware);

{ AMode emSoftware: Mesa's software device only. ADesc: what it opened. }
function I2MEglOpen(AMode: TI2MEglMode; out ADesc: string): Boolean;
function I2MEglMakeCurrent: Boolean;
procedure I2MEglClose;

{ Unix / X11: whether a double-buffered RGBA visual with a depth buffer is
  there (and ASamples-sample multisampling, if ASamples > 1). True elsewhere. }
function I2MGlxUsable(ASamples: Integer): Boolean;

implementation

uses
  SysUtils, StrUtils, dynlibs, ctypes
  {$IF DEFINED(UNIX) AND NOT DEFINED(DARWIN)}, x, xlib, xutil{$ENDIF};

type
  EGLDisplay = Pointer;
  EGLConfig = Pointer;
  EGLContext = Pointer;
  EGLSurface = Pointer;
  EGLDeviceEXT = Pointer;
  EGLint = cint32;
  PEGLint = ^EGLint;
  EGLBoolean = cuint32;

const
  EGL_NONE = $3038;
  EGL_SURFACE_TYPE = $3033;
  EGL_PBUFFER_BIT = $0001;
  EGL_RENDERABLE_TYPE = $3040;
  EGL_OPENGL_BIT = $0008;
  EGL_RED_SIZE = $3024;
  EGL_GREEN_SIZE = $3023;
  EGL_BLUE_SIZE = $3022;
  EGL_DEPTH_SIZE = $3025;
  EGL_OPENGL_API = $30A2;
  EGL_CONTEXT_MAJOR_VERSION = $3098;
  EGL_CONTEXT_MINOR_VERSION = $30FB;
  EGL_CONTEXT_OPENGL_PROFILE_MASK = $30FD;
  EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = $0001;
  EGL_PLATFORM_DEVICE_EXT = $313F;
  EGL_PLATFORM_SURFACELESS_MESA = $31DD;
  EGL_EXTENSIONS = $3055;
  EGL_VENDOR = $3053;

var
  Lib: TLibHandle = NilHandle;
  eglGetProcAddress: function(name: PChar): Pointer; cdecl;
  eglGetDisplay: function(id: Pointer): EGLDisplay; cdecl;
  eglInitialize: function(d: EGLDisplay; major, minor: PEGLint): EGLBoolean; cdecl;
  eglTerminate: function(d: EGLDisplay): EGLBoolean; cdecl;
  eglQueryString: function(d: EGLDisplay; name: EGLint): PChar; cdecl;
  eglChooseConfig: function(d: EGLDisplay; attribs: PEGLint; configs: Pointer;
    size: EGLint; num: PEGLint): EGLBoolean; cdecl;
  eglBindAPI: function(api: cuint): EGLBoolean; cdecl;
  eglCreateContext: function(d: EGLDisplay; c: EGLConfig; share: EGLContext;
    attribs: PEGLint): EGLContext; cdecl;
  eglDestroyContext: function(d: EGLDisplay; c: EGLContext): EGLBoolean; cdecl;
  eglMakeCurrent: function(d: EGLDisplay; draw, read: EGLSurface; c: EGLContext): EGLBoolean; cdecl;
  eglQueryDevicesEXT: function(max: EGLint; devices: Pointer; num: PEGLint): EGLBoolean; cdecl;
  eglQueryDeviceStringEXT: function(dev: EGLDeviceEXT; name: EGLint): PChar; cdecl;
  eglGetPlatformDisplayEXT: function(platform: cuint; native: Pointer; attribs: PEGLint): EGLDisplay; cdecl;

  Dpy: EGLDisplay = nil;
  Ctx: EGLContext = nil;

function LoadEgl: Boolean;

  function Get(const AName: string): Pointer;
  begin
    Result := GetProcAddress(Lib, AName);
    if (Result = nil) and Assigned(eglGetProcAddress) then Result := eglGetProcAddress(PChar(AName));
  end;

begin
  if Lib <> NilHandle then Exit(True);
  Lib := LoadLibrary('libEGL.so.1');
  if Lib = NilHandle then Lib := LoadLibrary('libEGL.so');
  if Lib = NilHandle then Exit(False);
  Pointer(eglGetProcAddress) := GetProcAddress(Lib, 'eglGetProcAddress');
  Pointer(eglGetDisplay) := Get('eglGetDisplay');
  Pointer(eglInitialize) := Get('eglInitialize');
  Pointer(eglTerminate) := Get('eglTerminate');
  Pointer(eglQueryString) := Get('eglQueryString');
  Pointer(eglChooseConfig) := Get('eglChooseConfig');
  Pointer(eglBindAPI) := Get('eglBindAPI');
  Pointer(eglCreateContext) := Get('eglCreateContext');
  Pointer(eglDestroyContext) := Get('eglDestroyContext');
  Pointer(eglMakeCurrent) := Get('eglMakeCurrent');
  Pointer(eglQueryDevicesEXT) := Get('eglQueryDevicesEXT');
  Pointer(eglQueryDeviceStringEXT) := Get('eglQueryDeviceStringEXT');
  Pointer(eglGetPlatformDisplayEXT) := Get('eglGetPlatformDisplayEXT');
  Result := Assigned(eglInitialize) and Assigned(eglChooseConfig) and
    Assigned(eglCreateContext) and Assigned(eglMakeCurrent) and Assigned(eglBindAPI);
end;

{ a 3.3 core context on this display, current with no surface }
function TryDisplay(ADisplay: EGLDisplay): Boolean;
const
  CfgAttr: array[0..10] of EGLint = (EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8, EGL_NONE);
  CtxAttr: array[0..6] of EGLint = (EGL_CONTEXT_MAJOR_VERSION, 3,
    EGL_CONTEXT_MINOR_VERSION, 3, EGL_CONTEXT_OPENGL_PROFILE_MASK,
    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE);
var
  ma, mi, n: EGLint;
  Cfg: EGLConfig;
  C: EGLContext;
begin
  Result := False;
  if ADisplay = nil then Exit;
  if eglInitialize(ADisplay, @ma, @mi) = 0 then Exit;
  n := 0;
  Cfg := nil;
  if (eglChooseConfig(ADisplay, @CfgAttr[0], @Cfg, 1, @n) = 0) or (n < 1) then
  begin   { a surfaceless-only display may have no pbuffer configs }
    if (eglChooseConfig(ADisplay, @CfgAttr[2], @Cfg, 1, @n) = 0) or (n < 1) then Cfg := nil;
  end;
  if eglBindAPI(EGL_OPENGL_API) = 0 then Exit;
  C := eglCreateContext(ADisplay, Cfg, nil, @CtxAttr[0]);
  if C = nil then Exit;
  if eglMakeCurrent(ADisplay, nil, nil, C) = 0 then
  begin
    eglDestroyContext(ADisplay, C);
    Exit;
  end;
  Dpy := ADisplay;
  Ctx := C;
  Result := True;
end;

function I2MEglOpen(AMode: TI2MEglMode; out ADesc: string): Boolean;
var
  Devs: array[0..15] of EGLDeviceEXT;
  n, i, pass: EGLint;
  Ext, Vendor: string;
  Soft: Boolean;
  D: EGLDisplay;
  P: PChar;
begin
  ADesc := '';
  if Ctx <> nil then Exit(I2MEglMakeCurrent);
  if not LoadEgl then
  begin
    ADesc := 'no libEGL';
    Exit(False);
  end;
  { the devices: hardware first, then software (pass 1) }
  if Assigned(eglQueryDevicesEXT) and Assigned(eglGetPlatformDisplayEXT) then
  begin
    n := 0;
    if eglQueryDevicesEXT(Length(Devs), @Devs[0], @n) <> 0 then
      for pass := 0 to 1 do
      begin
        if (AMode = emSoftware) and (pass = 0) then Continue;
        for i := 0 to n - 1 do
        begin
          Ext := '';
          if Assigned(eglQueryDeviceStringEXT) then
          begin
            P := eglQueryDeviceStringEXT(Devs[i], EGL_EXTENSIONS);
            if P <> nil then Ext := P;
          end;
          Soft := Pos('EGL_MESA_device_software', Ext) > 0;
          if Soft <> (pass = 1) then Continue;
          D := eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, Devs[i], nil);
          if TryDisplay(D) then
          begin
            Vendor := '';
            P := eglQueryString(D, EGL_VENDOR);
            if P <> nil then Vendor := P;
            ADesc := Format('EGL device %d (%s%s)', [i, Vendor, IfThen(Soft, ', software', '')]);
            Exit(True);
          end;
        end;
      end;
  end;
  { Mesa's surfaceless platform, then whatever the default display is }
  if Assigned(eglGetPlatformDisplayEXT) then
  begin
    D := eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, nil, nil);
    if TryDisplay(D) then
    begin
      ADesc := 'EGL surfaceless (Mesa)';
      Exit(True);
    end;
  end;
  if (AMode <> emSoftware) and Assigned(eglGetDisplay) and TryDisplay(eglGetDisplay(nil)) then
  begin
    ADesc := 'EGL default display';
    Exit(True);
  end;
  ADesc := 'no EGL device gave an OpenGL 3.3 core context';
  Result := False;
end;

function I2MEglMakeCurrent: Boolean;
begin
  Result := (Ctx <> nil) and (eglMakeCurrent(Dpy, nil, nil, Ctx) <> 0);
end;

procedure I2MEglClose;
begin
  if Ctx = nil then Exit;
  eglMakeCurrent(Dpy, nil, nil, nil);
  eglDestroyContext(Dpy, Ctx);
  Ctx := nil;
  Dpy := nil;
end;

function I2MGlxUsable(ASamples: Integer): Boolean;
{$IF DEFINED(UNIX) AND NOT DEFINED(DARWIN)}
const
  GLX_RGBA = 4;
  GLX_DOUBLEBUFFER = 5;
  GLX_DEPTH_SIZE = 12;
  GLX_SAMPLE_BUFFERS = 100000;
  GLX_SAMPLES = 100001;
var
  GL: TLibHandle;
  ChooseVisual: function(d: PDisplay; screen: cint; attribs: pcint): PXVisualInfo; cdecl;
  QueryExtension: function(d: PDisplay; errb, evb: pcint): cint; cdecl;
  D: PDisplay;
  V: PXVisualInfo;
  A: array[0..10] of cint;
  k, eb, vb: cint;
begin
  Result := False;
  D := XOpenDisplay(nil);
  if D = nil then Exit;   { not X11 (Wayland without Xwayland...): let the LCL try }
  try
    GL := LoadLibrary('libGL.so.1');
    if GL = NilHandle then GL := LoadLibrary('libGL.so');
    if GL = NilHandle then Exit;
    Pointer(ChooseVisual) := GetProcAddress(GL, 'glXChooseVisual');
    Pointer(QueryExtension) := GetProcAddress(GL, 'glXQueryExtension');
    if not Assigned(ChooseVisual) or not Assigned(QueryExtension) then Exit;
    if QueryExtension(D, @eb, @vb) = 0 then Exit;
    k := 0;
    A[k] := GLX_RGBA; Inc(k);
    A[k] := GLX_DOUBLEBUFFER; Inc(k);
    A[k] := GLX_DEPTH_SIZE; Inc(k);
    A[k] := 24; Inc(k);
    if ASamples > 1 then
    begin
      A[k] := GLX_SAMPLE_BUFFERS; Inc(k);
      A[k] := 1; Inc(k);
      A[k] := GLX_SAMPLES; Inc(k);
      A[k] := ASamples; Inc(k);
    end;
    A[k] := 0;
    V := ChooseVisual(D, DefaultScreen(D), @A[0]);
    Result := V <> nil;
    if V <> nil then XFree(V);
  finally
    XCloseDisplay(D);
  end;
end;
{$ELSE}
begin
  Result := True;
end;
{$ENDIF}

end.
