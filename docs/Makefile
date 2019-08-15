SPHINXOPTS  =
SPHINXBUILD = sphinx-build
SOURCEDIR   = _source
BUILDDIR    = _build

help:
	@$(SPHINXBUILD) --help

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -b $@ -c . "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)