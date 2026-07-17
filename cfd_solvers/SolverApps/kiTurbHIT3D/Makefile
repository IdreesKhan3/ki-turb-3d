###########################################################################
## KI-TURB isolated OpenLB application build

include definitions.mk

# The compilation controller passes these without mutating the OpenLB tree.
BUILD_DIR  ?= $(PWD)/build/kiturb
OUTPUT_PATH ?= $(PWD)/$(OUTPUT)
KITURB_PROFILE ?= serial

# Translate controller profiles to legacy OpenLB flags before global.mk.
ifeq ($(KITURB_PROFILE),openmp)
  PARALLEL_MODE := OMP
endif
ifeq ($(KITURB_PROFILE),mpi)
  PARALLEL_MODE := MPI
endif
ifeq ($(KITURB_PROFILE),mpi_openmp)
  PARALLEL_MODE := HYBRID
endif

include $(ROOT)/global.mk

BUILDTYPE := generic
LIBDIR    := build/generic/lib
DEPENDDIR := $(BUILD_DIR)/dep
OBJDIR    := $(BUILD_DIR)/obj
OUTPUT    := $(OUTPUT_PATH)

# HIT dynamics and FORCE descriptors are instantiated in this application.
CXXFLAGS  := $(filter-out -DOLB_PRECOMPILED,$(CXXFLAGS))
ifeq ($(KITURB_PROFILE),debug)
  override CXXFLAGS := $(filter-out -O%,$(CXXFLAGS)) -O0 -g3
endif
ifeq ($(KITURB_PROFILE),sanitizer)
  override CXXFLAGS := $(filter-out -O%,$(CXXFLAGS)) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
  override LDFLAGS := -fsanitize=address,undefined $(LDFLAGS)
endif

OBJECTS := $(foreach file,$(SRC),$(OBJDIR)/$(file:.cpp=.o))
DEPS    := $(foreach file,$(SRC),$(DEPENDDIR)/$(file:.cpp=.d))

.PHONY: all depend compile link clean cleanobj cleandep capabilities
all: depend compile link

$(OBJDIR) $(DEPENDDIR) $(dir $(OUTPUT)):
	@mkdir -p $@

depend: $(DEPS)

$(DEPENDDIR)/%.d: %.cpp | $(DEPENDDIR)
	@echo Create dependencies for $<
	@$(SHELL) -ec '$(CXX) -M $(CXXFLAGS) $(IDIR) $< | sed -e "s!$*\.o!$(OBJDIR)/$*\.o!1" > $@'

compile: $(OBJECTS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	@echo Compile $<
	$(CXX) $(CXXFLAGS) $(IDIR) -c $< -o $@

clean: cleanobj cleandep
	@rm -f $(OUTPUT)

cleanobj:
	@rm -rf $(OBJDIR)

cleandep:
	@rm -rf $(DEPENDDIR)

$(ROOT)/$(LIBDIR)/lib$(LIB).a:
	@cd $(ROOT) && $(MAKE) all

link: $(OUTPUT)

$(OUTPUT): $(OBJECTS) $(ROOT)/$(LIBDIR)/lib$(LIB).a | $(dir $(OUTPUT))
	@echo Link $@
	$(CXX) $(OBJECTS) $(LDFLAGS) -L$(ROOT)/$(LIBDIR) -l$(LIB) -lz -o $@

capabilities: $(OUTPUT)
	$(OUTPUT) --capabilities

ifneq "$(strip $(wildcard $(DEPS)))" ""
  include $(DEPS)
endif
