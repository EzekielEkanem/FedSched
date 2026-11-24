
# ============================================
# FedSched Makefile
# Target: Linux ARM64
# ============================================

CLANG   := clang
CC      := cc
BPFTOOL := bpftool

# Project paths
BPF_SRC  := src/bpf/fedsched.bpf.c
USER_SRC := src/user/fedsched.c
USER_CONTROLLER := src/user/controller.h
COMMON_SRC := src/common/fedsched.h
BUILD    := build

BPF_OBJ  := $(BUILD)/fedsched.bpf.o
BPF_SKEL := $(BUILD)/fedsched.skel.h
TARGET   := $(BUILD)/fedsched

# Adjust this if vmlinux.h is elsewhere
VMLINUX_DIR ?= src/bpf

# Existing libbpf Bootstrap headers
LIBBPF_DIR := $(HOME)/libbpf-bootstrap/libbpf/src

# Compiler flags
BPF_CFLAGS := -g -O2 -target bpf \
              -D__TARGET_ARCH_arm64 \
              -Wno-missing-declarations \
              -I$(VMLINUX_DIR) \
              -I$(LIBBPF_DIR) \
			  -Isrc/common

USER_CFLAGS := -g -O2 -I$(BUILD) \
               -Isrc/common
USER_LIBS   := $(shell pkg-config --cflags --libs libbpf) -lelf -lz

# Default build target
.PHONY: all clean run rebuild

all: $(TARGET)

# Create build directory
$(BUILD):
	mkdir -p $(BUILD)

# Compile eBPF source
$(BPF_OBJ): $(BPF_SRC) src/common/fedsched.h $(VMLINUX_DIR)/vmlinux.h | $(BUILD)
	$(CLANG) $(BPF_CFLAGS) -c $(BPF_SRC) -o $(BPF_OBJ)

# Generate libbpf skeleton
$(BPF_SKEL): $(BPF_OBJ)
	$(BPFTOOL) gen skeleton $(BPF_OBJ) > $(BPF_SKEL)

# Compile userspace loader
$(TARGET): $(USER_SRC) $(BPF_SKEL) $(COMMON_SRC) $(USER_CONTROLLER)
	$(CC) $(USER_CFLAGS) $(USER_SRC) \
	    -o $(TARGET) $(USER_LIBS)

# Run FedSched (requires privileges)
run: all
	sudo ./$(TARGET)

# Delete generated build files
clean:
	rm -rf $(BUILD)

# Force a complete rebuild
rebuild: clean all
