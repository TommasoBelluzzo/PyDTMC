SPHINX_EXE  = python -m sphinx
SOURCE_DIR  = _source
BUILD_DIR   = _build
SPHINX_OPTS =

help:
	@$(SPHINX_EXE) --help

.PHONY: help Makefile

%: Makefile
	@rm -rf "$(BUILD_DIR)/$@/*"
	@$(SPHINX_EXE) -n -b $@ -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/$@" $(O)
	@$(SPHINX_EXE) -n -b coverage -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/$@" $(O)