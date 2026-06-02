# Makefile — build baobzi_mex.mexa64 without needing a MATLAB license.
#
# Usage:
#   make                     # build baobzi_mex.mexa64 from the committed gateway
#   make mwrap_gen           # regenerate baobzi_mex_gen.cpp + bz_*.m from baobzi.mw
#   make clean
#
# Override variables as needed:
#   make MATLAB_ROOT=/path/to/matlab BAOBZI_BUILD=../../build MWRAP=/path/to/mwrap
#
# baobzi_mex_gen.cpp and bz_*.m are mwrap-generated but *committed*, so a plain
# `make` needs neither mwrap nor flex/bison. They are only regenerated when
# baobzi.mw is newer (or via `make mwrap_gen`); mwrap defaults to PATH.

MATLAB_ROOT  ?= /usr/local/MATLAB/R2025b
BAOBZI_BUILD ?= ../../build
BAOBZI_INC   ?= ../../include
MWRAP        ?= mwrap

MATLAB_INC   := $(MATLAB_ROOT)/extern/include
MATLAB_LIB   := $(MATLAB_ROOT)/bin/glnxa64
MATLAB_MAP   := $(MATLAB_ROOT)/extern/lib/glnxa64/mexFunction.map

CXX      ?= g++
# -DR2008OO selects mwrap's classdef/mwptr object-pointer path in the gateway.
CXXFLAGS := -std=c++17 -O2 -fPIC -DMATLAB_MEX_FILE -DR2008OO \
            -I$(MATLAB_INC) -I$(BAOBZI_INC)
# -static-libstdc++/-static-libgcc so the .mexa64 doesn't depend on a newer
# GLIBCXX than MATLAB's bundled libstdc++.
LDFLAGS  := -shared \
            -L$(MATLAB_LIB) -lmex -lmx \
            -Wl,-rpath,$(MATLAB_LIB) \
            -Wl,--version-script,$(MATLAB_MAP) \
            -static-libstdc++ -static-libgcc

all: baobzi_mex.mexa64

baobzi_mex.mexa64: baobzi_mex_gen.cpp $(BAOBZI_BUILD)/libbaobzi_c.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

mwrap_gen: baobzi_mex_gen.cpp

# Regenerate the committed gateway + stubs only when the spec changes.
baobzi_mex_gen.cpp: baobzi.mw
	$(MWRAP) -mex baobzi_mex -c baobzi_mex_gen.cpp -mb baobzi.mw

clean:
	rm -f baobzi_mex.mexa64

.PHONY: all mwrap_gen clean
