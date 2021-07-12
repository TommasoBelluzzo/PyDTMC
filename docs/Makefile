SPHINX_EXE:=python -m sphinx
SOURCE_DIR:=source
BUILD_DIR:=build

help:
	@$(SPHINX_EXE) --help

.PHONY: help Makefile

%: Makefile
	@rm -rf $(BUILD_DIR)/$@/*
	@$(SPHINX_EXE) -b $@ $(SOURCE_DIR)/ $(BUILD_DIR)/$@/ $(SPHINX_OPTS)