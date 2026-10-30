# Release builder for the v2m GUI (after mcx's .github/linux_release_builder,
# which builds mcxstudio the same way).
#
# manylinux2014 is CentOS 7, glibc 2.17: linked here v2m needs at most
# GLIBC_2.17 and runs on any distribution since (Ubuntu 14.04 on). Built on the
# ubuntu-22.04 runner it picked up GLIBC_2.34 symbols (__libc_start_main, and
# dlopen and the rest, which moved into libc in 2.34) and would not start on
# Ubuntu 20.04 or older. gtk2, glib, pango, cairo, X11 and libGL are the
# system's, by the sonames every distribution has.
#
# Run with "docker run", not as a job container: the JavaScript actions
# (checkout, upload-artifact) need node 20, which needs glibc 2.28.

FROM quay.io/pypa/manylinux2014_x86_64

# Free Pascal 3.2.2 and Lazarus 2.2.0 (the version v2m's form files are kept
# at, as mcxstudio's: a newer IDE rewrites them), for the gtk2 widget set.
# lazbuild compiles the LCL into the Lazarus tree on first use, so the tree is
# left writable for a build run as the caller
RUN yum install -y gtk2-devel mesa-libGL-devel libX11-devel wget which make \
    && yum clean all \
    && wget -q -O /tmp/fpc.tar https://downloads.sourceforge.net/project/freepascal/Linux/3.2.2/fpc-3.2.2.x86_64-linux.tar \
    && cd /tmp && tar xf fpc.tar && cd fpc-3.2.2.x86_64-linux \
    && printf '/usr/local\nn\nn\nn\n' | ./install.sh \
    && cd / && rm -rf /tmp/fpc* \
    && wget -q -O /tmp/lazarus.tar.gz "https://downloads.sourceforge.net/project/lazarus/Lazarus%20Zip%20_%20GZip/Lazarus%202.2.0/lazarus-2.2.0-0.tar.gz" \
    && tar xzf /tmp/lazarus.tar.gz -C /opt && rm -f /tmp/lazarus.tar.gz \
    && cd /opt/lazarus && make lazbuild \
    && ln -s /opt/lazarus/lazbuild /usr/local/bin/lazbuild \
    && chmod -R a+rwX /opt/lazarus

WORKDIR /src
