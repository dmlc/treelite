ifndef LINT_LANG
	LINT_LANG="all"
endif

ifndef NPROC
	NPROC=1
endif

doxygen:
	cd docs; doxygen

cpp-coverage:
	rm -rf build; mkdir build; cd build; cmake .. -DTEST_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug && make -j$(NPROC)

all:
	rm -rf build; mkdir build; cd build; cmake .. && make -j$(NPROC)

pippack:
	cd python && python setup.py sdist && mv dist/*.tar.gz ..
