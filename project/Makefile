INSTALL_DIR ?= /usr/local/

# Location of the CUDA Toolkit
CUDA_INSTALL_DIR ?= /usr/local/cuda-10.2
TCNN_INSTALL_DIR ?= ..

################################################################################
TARGET := meshbox


CXXFLAGS := -fPIC -Wall -Wextra
DEFINES := -DTCNN_MIN_GPU_ARCH=70 -std=c++14

INCLUDES  := \
	-I include \
	-I${CUDA_INSTALL_DIR}/include \
	-I${TCNN_INSTALL_DIR}/include \
	-I${TCNN_INSTALL_DIR}/dependencies \
	-I${TCNN_INSTALL_DIR}/dependencies/cutlass/include \
	-I${TCNN_INSTALL_DIR}/dependencies/cutlass/tools/util/include \
	-I${TCNN_INSTALL_DIR}/dependencies/fmt/include \
	-I${TCNN_INSTALL_DIR}/dependencies/eigen


LIBRARIES := \
	-L${TCNN_INSTALL_DIR}/build -ltiny-cuda-nn \
	-L${TCNN_INSTALL_DIR}/build/dependencies/fmt -lfmt \
 	-L${CUDA_INSTALL_DIR}/lib64 -lcudart_static -lcuda \
 	-lrt -ldl -lpthread -lm

SOURCE := main.cpp \
	src/mesh_camera.cu \
	src/mesh_image.cu \
	src/mesh_point.cu \
	../dependencies/stbi/stbi_wrapper.cpp

APPLETS := $(shell ./makebox --source)
SOURCE += $(APPLETS)

################################################################################
TARGET_SIZE := 64

NVCC := $(CUDA_INSTALL_DIR)/bin/nvcc

# internal flags
CCFLAGS     :=
LDFLAGS     := -fPIC -O3

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
endif

NVCCFLAGS := \
	-forward-unknown-to-host-compiler \
	-Xcompiler=-mf16c \
	-Xcompiler=-Wno-float-conversion \
	-Xcompiler=-fno-strict-aliasing \
	-Xcudafe=--diag_suppress=unrecognized_gcc_pragma \
	--extended-lambda \
	--expt-relaxed-constexpr 

SMS ?= 75


ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


CXXFLAGS := ${CXXFLAGS} ${DEFINES}
NVCCFLAGS := ${NVCCFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))


#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $< -o $@

%.o : %.c
	${CC} ${CFLAGS} ${INCS} -c $< -o $@


#****************************************************************************
# Target rules
#****************************************************************************
all: ${TARGET}

${TARGET}: applets.h ${OBJECT}
	${CXX} -o ${TARGET} ${LDFLAGS} ${OBJECT} ${LIBRARIES}

applets.h: busybox.config
	./makebox --table > applets.h

install:
	#${STRIP} ${TARGET}
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

clean:
	rm -f ${OBJECT} ${TARGET}

