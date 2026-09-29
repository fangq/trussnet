# SPDX-License-Identifier: GPL-3.0-or-later
# v2m - a graphical front end for v2mesh (Lazarus / FPC).
# The Lazarus-directory detection below is mcxstudio2's (same author, GPL-3.0+).
#
# lazbuild wants to compile the Lazarus packages we depend on into their own
# install directories, which under a distro package live in /usr and are not
# writable.  Giving it a project-local primary config path redirects that
# output here instead, so a plain "make" works for an ordinary user and in CI
# without touching the system Lazarus or the developer's own IDE config.

BINARY    ?= v2m
WIDGETSET ?= gtk2
PCP       ?= .lazpcp
LAZBUILD  ?= lazbuild

# Where lazbuild's own LCL lives.
#
# lazbuild normally reads this out of the config its distribution ships.  A
# machine without that file -- a partial install, a Lazarus that has never
# been run, a CI image -- gets "invalid Lazarus directory: " and stops, the
# empty string being what it had to go on.  Naming the directory here is what
# an IDE would have written into the config.
#
# Found by looking for the lcl directory inside a candidate rather than for
# the candidate itself.  /usr/share/lazarus is a real path on a machine that
# keeps its versions underneath it -- /usr/share/lazarus/4.2.0 -- and naming
# that parent produces the very error this is here to avoid.
#
# The install $(LAZBUILD) itself lives in is tried first, so the LCL always
# matches the tool that is about to use it.  That is also how a machine with
# several Lazarus versions picks between them:
#
#     make LAZBUILD=/usr/lib/lazarus/2.2.0/lazbuild
#
# and how any layout this does not know about is handled, since lazbuild is
# on PATH wherever it was installed.  Failing that, the newest of whatever is
# in the usual places, by version rather than by name so that 10.0 sorts above
# 2.2.  Failing that too, nothing is passed and lazbuild is left to its own
# config, which is what it had before any of this.
#
# Either half can be overridden:  make LAZDIR=/opt/lazarus
LAZROOTS ?= /usr/lib/lazarus /usr/lib/lazarus/* \
            /usr/lib64/lazarus /usr/lib64/lazarus/* \
            /usr/share/lazarus /usr/share/lazarus/* \
            /usr/local/share/lazarus /usr/local/share/lazarus/* \
            /usr/local/lib/lazarus /usr/local/lib/lazarus/* \
            /opt/lazarus /opt/lazarus/* /opt/lazarus-* \
            $(HOME)/lazarus $(HOME)/fpcupdeluxe/lazarus \
            /Applications/Lazarus /c/lazarus C:/lazarus

# A case statement would read better than the test below, but make ends a
# $(shell ...) at the first unmatched ')' and "in /*)" is one -- so the leading
# slash is tested by stripping it and seeing whether anything changed.

# -V is what puts 10.0 above 2.2; it is not in every sort, so ask once.
LAZSORT := $(shell printf 'a\n' | sort -V >/dev/null 2>&1 \
                   && echo 'sort -Vr' || echo 'sort -r')

ifeq ($(origin LAZDIR),undefined)
LAZDIR := $(shell \
    p=$$(command -v $(LAZBUILD) 2>/dev/null); n=0; \
    while [ -n "$$p" ] && [ -L "$$p" ] && [ $$n -lt 20 ]; do \
        n=$$((n + 1)); \
        l=$$(readlink "$$p"); \
        if [ "$${l#/}" != "$$l" ]; then p=$$l; \
        else p=$$(dirname "$$p")/$$l; fi; \
    done; \
    if [ -n "$$p" ]; then \
        p=$$(cd "$$(dirname "$$p")" 2>/dev/null && pwd -P); \
    fi; \
    for d in "$$LAZARUSDIR" "$$p" \
             $$(ls -d $(LAZROOTS) 2>/dev/null | $(LAZSORT)); do \
        if [ -n "$$d" ] && [ -d "$$d/lcl" ]; then printf '%s' "$$d"; exit 0; fi; \
    done)
endif

LAZFLAGS = --pcp=$(PCP) --ws=$(WIDGETSET) \
           $(if $(LAZDIR),--lazarusdir=$(LAZDIR))

PREFIX ?= /usr/local

.PHONY: all debug forms icons clean distclean install uninstall

all: forms
	$(LAZBUILD) $(LAZFLAGS) --build-mode=Release $(BINARY).lpi

debug: forms
	$(LAZBUILD) $(LAZFLAGS) --build-mode=Debug $(BINARY).lpi

# lazbuild decides what to recompile by the .pas timestamps, so a form edited on
# its own (.lfm newer than its unit) would be skipped: touch the unit.
forms:
	@for f in *.lfm; do \
	    u=$${f%.lfm}.pas; \
	    if [ -f "$$u" ] && [ "$$f" -nt "$$u" ]; then touch "$$u"; fi; \
	done

# the toolbar icons (MCX Studio's set) as a Lazarus resource include; the .lrs
# is committed, so only a changed icon needs this
icons:
	lazres i2micons.lrs icons/*.png

install: all
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 bin/$(BINARY) $(DESTDIR)$(PREFIX)/bin/$(BINARY)

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(BINARY)

clean:
	rm -rf lib bin debug

distclean: clean
	rm -rf $(PCP)
