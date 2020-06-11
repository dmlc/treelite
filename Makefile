ifndef LINT_LANG
	LINT_LANG="all"
endif

ifndef NPROC
	NPROC=1
endif
lint:
	wget -nc https://raw.githubusercontent.com/dmlc/dmlc-core/9db4b20c868341abe2a9fe52b652f7d9447ed406/scripts/lint.py
	PYTHONPATH=./python:./runtime/python python lint.py treelite $(LINT_LANG) include src python tests/python \
		--exclude_path python/treelite/gallery/sklearn --pylint-rc $(PWD)/python/.pylintrc

doxygen:
	cd docs; doxygen

cpp-coverage:
	rm -rf build; mkdir build; cd build; cmake .. -DTEST_COVERAGE=ON -DENABLE_PROTOBUF=ON -DCMAKE_BUILD_TYPE=Debug && make -j$(NPROC)

all:
	rm -rf build; mkdir build; cd build; cmake .. -DENABLE_PROTOBUF=ON && make -j$(NPROC)

pippack:
	cd python && python setup.py sdist && mv dist/*.tar.gz .. && cd ../runtime/python && python setup.py sdist && mv dist/*.tar.gz ../..
