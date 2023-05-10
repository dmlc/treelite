ifndef LINT_LANG
	LINT_LANG="all"
endif

ifndef NPROC
	NPROC=1
endif
lint:
	PYTHONPATH=./python:./runtime/python python tests/ci_build/lint.py treelite $(LINT_LANG) include src python tests/python \
		--pylint-rc $(PWD)/python/.pylintrc

doxygen:
	cd docs; doxygen

cpp-coverage:
	rm -rf build; mkdir build; cd build; cmake .. -DTEST_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug && make -j$(NPROC)

all:
	rm -rf build; mkdir build; cd build; cmake .. && make -j$(NPROC)

pippack:
	cd python && python setup.py sdist && mv dist/*.tar.gz ..
