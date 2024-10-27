TARGETS = echo
CXXFILES = 

BITS ?= 64
ODIR := ./obj$(BITS)

output_folder := $(shell mkdir -p $(ODIR))

COMMONOFILES = $(patsubst %, $(ODIR)/%.o, $(CXXFILES))
ALLOFILES = $(patsubst %, $(ODIR)/%.o, $(CXXFILES) $(TARGETS))
EXEFILES = $(patsubst %, $(ODIR)/%.exe, $(TARGETS))

DFILES = $(patsubst %.o, %.d, $(ALLOFILES))

CXX = g++
LD = g++
CXXFLAGS = -MMD -O3 -m$(BITS) -ggdb -std=c++17 -Wall -Werror
LDFLAGS = -m$(BITS) -lpthread -lcrypto

.DEFAULT_GOAL = all
.PRECIOUS: $(ALLOFILES)
.PHONY: all clean

all: $(EXEFILES)


$(ODIR)/%.o: %.cc
	@echo "[CXX] $< --> $@"
	@$(CXX) $< -o $@ -c $(CXXFLAGS)

$(ODIR)/%.exe: $(ODIR)/%.o $(COMMONOFILES)
	@echo "[LD] $^ --> $@"
	@$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	@echo Cleaning up...
	@rm -rf $(ODIR)

-include $(DFILES)


