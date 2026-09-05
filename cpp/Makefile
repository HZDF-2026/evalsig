# evalsig C++17 port — bit-exact with the Python/Go references.
# make / make all   build ../dist/cpp/evalsig(.exe)
# make test         build and run the golden-table test against tests/refdata/core.json
# make clean        remove build artifacts
#
# Directory commands are shell-guarded because mingw32-make without a POSIX
# shell cannot exec `mkdir`/`rm` directly; `cmd /c` is a real executable, so it
# works through CreateProcess as well as through any sh. Backslash path
# rewriting is Windows-only: on POSIX a literal `..\dist\cpp` is one oddly
# named directory and the link step dies on a missing output path.

CXX ?= g++
CXXSTD ?= -std=c++17
WARN ?= -Wall -Wextra -O2
# The golden tables are bit-exact with CPython, which rounds x*y and x+z
# separately. clang (notably on arm64) contracts a*b+c into fma by default and
# flips the last bit; g++ on x86-64 happens not to, but pass the flag anyway
# so every toolchain keeps the same non-fused semantics.
FPMATH := -ffp-contract=off
CXXFLAGS := $(CXXSTD) $(WARN) $(FPMATH)
BUILD := build
DIST := ../dist/cpp

SRCS := pynum.cpp pyjson.cpp intervals.cpp power.cpp variance.cpp decide.cpp \
        sequential.cpp cli.cpp main.cpp
LIB_SRCS := pynum.cpp pyjson.cpp intervals.cpp power.cpp variance.cpp decide.cpp \
            sequential.cpp cli.cpp
OBJS := $(addprefix $(BUILD)/,$(SRCS:.cpp=.o))
LIB_OBJS := $(addprefix $(BUILD)/,$(LIB_SRCS:.cpp=.o))

BIN := $(DIST)/evalsig$(if $(filter $(OS),Windows_NT),.exe,)
TEST_BIN := $(BUILD)/test_refdata$(if $(filter $(OS),Windows_NT),.exe,)

ifeq ($(OS),Windows_NT)
MD := cmd /d /c md
RMRF := cmd /d /c rmdir /s /q
FIXPATH = $(subst /,\,$1)
else
MD := mkdir -p
RMRF := rm -rf
FIXPATH = $1
endif

.PHONY: all test clean

all: $(BIN)

$(BIN): $(OBJS) | $(DIST)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(DIST) $(BUILD):
	-$(MD) $(call FIXPATH,$@)

$(BUILD)/%.o: %.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: $(BIN) $(TEST_BIN)
	$(TEST_BIN)

$(TEST_BIN): test_refdata.cpp $(LIB_OBJS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	-$(RMRF) $(call FIXPATH,$(BUILD))
	-$(RMRF) $(call FIXPATH,$(DIST))
