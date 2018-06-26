ifndef LINT_LANG
	LINT_LANG="all"
endif

ifndef NPROC
	NPROC=1
endif
lint: 
	python3 dmlc-core/scripts/lint.py treelite ${LINT_LANG} include src python \
	--exclude_path python/treelite/gallery/sklearn

doxygen:
	cd docs; doxygen

all:
	rm -rf build; mkdir build; cd build; cmake .. && make -j${NPROC}
