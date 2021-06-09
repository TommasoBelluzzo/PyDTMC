SPHINX_EXE  = sphinx-build
SOURCE_DIR  = _source
BUILD_DIR   = _build
SPHINX_OPTS =

help:
	@$(SPHINX_EXE) --help

.PHONY: help Makefile

%: Makefile
	@rm -rf "$(BUILD_DIR)/$@/*"
	@$(SPHINX_EXE) -nW -b $@ -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/$@" $(O)
	@$(SPHINX_EXE) -nW -b coverage -c . "$(SOURCE_DIR)" "$(BUILD_DIR)/$@" $(O)