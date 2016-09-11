UNAME := $(shell uname -s)
OBJDIRR=.build-release
OBJDIRD=.build-debug

ifeq ($(UNAME),Linux)
    PLATFORM=Linux
    PKG  =pkg-config
    MOC  =moc
    CUDA_DIR := /usr/local/cuda
    LIBS = -L/usr/lib64 -L$(CUDA_DIR)/lib64 -lhdf5_hl -lm -lopenblas
    INC = -I/usr/include -I$(CUDA_DIR)/include
    A =.a
    LIB_PREFIX =lib
endif

ifeq ($(UNAME),Darwin)
    PLATFORM=Darwin
    PKG  =/usr/local/bin/pkg-config
    MOC  =/usr/local/Cellar/qt/4.8.6/bin/moc
    LIBS = -framework Accelerate
    INC += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers/
endif

CC=gcc
LINK=gcc
AR=ar r
AR_OUT=
LINK_OUT= -o
LINK_OPT=
MINUS_O = -o
INC += -Iinclude
CFLAGS   = -std=c++11 -Wall -DUSE_OPENCV -Wno-unused-variable -Wno-unused-function -fPIC -g -O3 $(INC)
CFLAGSD  = -std=c++11 -Wall -DUSE_OPENCV -Wno-unused-variable -Wno-unused-function -fPIC -g -DDEBUG $(INC)
LIBS    += -lstdc++ -llmdb -lboost_system-mt -lboost_thread-mt -lboost_filesystem -lboost_regex -lboost_iostreams-mt -ljsoncpp -lGL -lGLU
LIBS    += `$(PKG) --libs libglog`
LIBSQT  += `$(PKG) --libs QtGui QtOpenGL`
LIBSD    = $(LIBS)
LIBSQTD  = $(LIBSQT)
DEPENDS= -MMD -MF $@.dep

EVERY_BIN=viz-r$(EXE) viz-d$(EXE)

UTIL = viz/miniutils.cpp
VIZ  = viz/viz.cpp viz/viz-progress.cpp
TSNE = t-sne/tsne.cpp t-sne/sptree.cpp

UTIL_R = $(patsubst %.cpp, $(OBJDIRR)/%.o, $(UTIL))
UTIL_D = $(patsubst %.cpp, $(OBJDIRD)/%.o, $(UTIL))
VIZ_R = $(patsubst %.cpp, $(OBJDIRR)/%.o, $(VIZ))
VIZ_D = $(patsubst %.cpp, $(OBJDIRD)/%.o, $(VIZ))
TSNE_R = $(patsubst %.cpp, $(OBJDIRR)/%.o, $(TSNE))
TSNE_D = $(patsubst %.cpp, $(OBJDIRD)/%.o, $(TSNE))

EVERY_OBJ = $(VIZ_R) $(VIZ_D) $(UTIL_R) $(UTIL_D) $(TSNE_R) $(TSNE_D)
DEP = $(patsubst %.o,%.o.dep, $(EVERY_OBJ))

all: dirs $(EVERY_BIN)

$(OBJDIRR)/viz/viz.o: viz/../.generated/viz.moc
viz/../.generated/viz.moc: viz/viz.cpp
	$(MOC) -o $@ $<

viz-r$(EXE): $(VIZ_R) $(UTIL_R) $(TSNE_R)
	$(LINK) $(LINK_OPT) $(LINK_OUT)$@ $^ $(LIBS) $(LIBSQT)
viz-d$(EXE): $(VIZ_D) $(UTIL_D) $(TSNE_D)
	$(LINK) $(LINK_OPT) $(LINK_OUT)$@ $^ $(LIBSD) $(LIBSQTD)

$(OBJDIRR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c $<  $(MINUS_O)$@ $(DEPENDS)
$(OBJDIRD)/%.o: %.cpp
	$(CC) $(CFLAGSD) -c $<  $(MINUS_O)$@ $(DEPENDS)

.PHONY: depend clean dirs

clean:
	$(RM) $(EVERY_BIN) $(EVERY_OBJ) .generated/*.moc *.ilk *.pdb $(DEP)
	rm -rf .generated
	rm -rf $(OBJDIRD)
	rm -rf $(OBJDIRR)

depends:
	cat  $(DEP) > Makefile.dep

# build directories
.generated:
	mkdir -p .generated
$(OBJDIRR)/viz:
	mkdir -p $@
$(OBJDIRD)/viz:
	mkdir -p $@
$(OBJDIRR)/t-sne:
	mkdir -p $@
$(OBJDIRD)/t-sne:
	mkdir -p $@

dirs: .generated $(OBJDIRR)/viz $(OBJDIRD)/viz $(OBJDIRR)/t-sne $(OBJDIRD)/t-sne

-include Makefile.dep
