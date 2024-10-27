BITS=64

# Output directory
ODIR = obj32
$(shell mkdir -p $(ODIR))

# Compiler configuration and flags
CXX = g++
# CXX = clang++
# CXXFLAGS = -MMD -ggdb -O3 -std=c++20 -m$(BITS)
# CXXFLAGS = -MMD -ggdb -fgnu-tm -gdwarf-5 -O3 -std=c++20 -m$(BITS)
# CXXFLAGS = -fgnu-tm -O3 -mrtm -std=c++20 -m$(BITS)
# LDFLAGS = -m$(BITS) -Loneapi-tbb-2021.8.0/lib/intel64/gcc4.8 -ltbb -lpthread -lrt

CXXFLAGS = -MMD -ggdb -O3 -std=gnu++20 -m$(BITS) -fgnu-tm
LDFLAGS	 = -m$(BITS) -Loneapi-tbb-2021.8.0/lib/intel64/gcc4.8 -ltbb -lpthread -lrt -fgnu-tm

# The executable we will build
TARGET = $(ODIR)/main

# Source files
CXXFILES = main

# Create the .o names from the CXXFILES
OFILES = $(patsubst %, $(ODIR)/%.o, $(CXXFILES))

# Create .d files to store dependency information, so that we don't need to
# clean every time before running make
DFILES = $(patsubst %.o, %.d, $(OFILES))

# Default rule builds the executable
all: $(TARGET)

# clean up everything by clobbering the output folder
clean:
	@echo cleaning up...
	@rm -rf $(ODIR)

run: $(TARGET)
	./$(TARGET)

# build an .o file from a .cpp file
$(ODIR)/%.o: %.cpp
	@echo [CXX] $< "-->" $@
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

# Link rule for building the target from .o files
$(TARGET): $(OFILES)
	@echo [LD] $^ "-->" $@
	@$(CXX) -o $@ $^ $(LDFLAGS)

# Remember that 'all' and 'clean' aren't real targets
.PHONY: all clean

# Pull in all dependencies
-include $(DFILES)
