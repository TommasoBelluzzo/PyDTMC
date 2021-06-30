SHELL:=/bin/bash

SPHINX_EXE:=python -m sphinx
SOURCE_DIR:=source
BUILD_DIR:=build

help:
	@$(SPHINX_EXE) --help

.PHONY: help Makefile

%: Makefile
	@if [[ $@ == 'html' ]]; then\
		SPHINX_OPTS=-nW -t html_prolog;\
	elif [[ $@ = 'latex' ]]; then\
		SPHINX_OPTS=-nW -t latex_prolog;\
	else\
		SPHINX_OPTS=-nW;\
	fi
	@rm -rf $(BUILD_DIR)/doctest/*
	@rm -rf $(BUILD_DIR)/linkcheck/*
	@rm -rf $(BUILD_DIR)/coverage/*
	@rm -rf $(BUILD_DIR)/$@/*
	@$(SPHINX_EXE) -b doctest $(SOURCE_DIR) $(BUILD_DIR)/doctest $(SPHINX_OPTS) $(O)
	@$(SPHINX_EXE) -b linkcheck $(SOURCE_DIR) $(BUILD_DIR)/linkcheck $(SPHINX_OPTS) $(O)
	@$(SPHINX_EXE) -b coverage $(SOURCE_DIR) $(BUILD_DIR)/coverage $(SPHINX_OPTS) $(O)
	@$(SPHINX_EXE) -b $@ $(SOURCE_DIR) $(BUILD_DIR)/$@ $(SPHINX_OPTS) $(O)