# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Project root (where pyproject.toml lives)
PROJECT_ROOT  = ..

# uv run with docs dependencies
UVRUN         = cd $(PROJECT_ROOT) && uv run --extra docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(UVRUN) $(SPHINXBUILD) -M help "docs" "docs/$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile doxygen clean html

# Run Doxygen to generate XML for Breathe
doxygen:
	@echo "Running Doxygen..."
	@mkdir -p _doxygen
	@doxygen Doxyfile

# Clean build artifacts
clean:
	@rm -rf $(BUILDDIR)
	@rm -rf _doxygen

# Build HTML with Doxygen first, using uv for dependencies
html: doxygen
	$(UVRUN) $(SPHINXBUILD) -M html "docs" "docs/$(BUILDDIR)" $(SPHINXOPTS) $(O)
