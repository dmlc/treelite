ifndef LINT_LANG
	LINT_LANG="all"
endif

ifndef NPROC
	NPROC=1
endif
lint:
	python3 dmlc-core/scripts/lint.py treelite $(LINT_LANG) include src python \
	--exclude_path python/treelite/gallery/sklearn --pylint-rc $(PWD)/python/.pylintrc

doxygen:
	cd docs; doxygen

cpp-coverage:
	rm -rf build; mkdir build; cd build; cmake .. -DTEST_COVERAGE=ON -DENABLE_PROTOBUF=ON -DCMAKE_BUILD_TYPE=Debug && make -j$(NPROC)

all:
	rm -rf build; mkdir build; cd build; cmake .. -DENABLE_PROTOBUF=ON && make -j$(NPROC)

pippack:
	rm -rfv build
	rm -rfv python/build python/dist python/treelite.egg-info python/treelite/treelite_runtime.zip python/treelite/libtreelite*.so
	rm -rfv treelite-python
	mkdir treelite-python
	cp python/pip_util/setup_pip.py treelite-python/setup.py
	cp python/pip_util/MANIFEST.in treelite-python/MANIFEST.in
	cp -r python/treelite/ treelite-python
	cp CMakeLists.txt treelite-python/treelite/
	cp VERSION treelite-python/treelite/
	cp -r cmake treelite-python/treelite/
	cp -r dmlc-core treelite-python/treelite/
	cp -r 3rdparty treelite-python/treelite/
	cp -r include treelite-python/treelite/
	cp -r runtime treelite-python/treelite/
	cp -r src treelite-python/treelite/
	mkdir treelite-python/treelite/runtime/native/python/treelite_runtime/common/
	mkdir treelite-python/treelite/runtime/native/src/common/
	cp include/treelite/c_api_common.h treelite-python/treelite/runtime/native/include/treelite/
	cp include/treelite/logging.h treelite-python/treelite/runtime/native/include/treelite/
	cp src/c_api/c_api_common.cc treelite-python/treelite/runtime/native/src/c_api/
	cp src/c_api/c_api_error.cc treelite-python/treelite/runtime/native/src/c_api/
	cp src/c_api/c_api_error.h treelite-python/treelite/runtime/native/src/c_api/
	cp src/common/math.h treelite-python/treelite/runtime/native/src/common/
	cp src/common/filesystem.h treelite-python/treelite/runtime/native/src/common/
	cp src/logging.cc treelite-python/treelite/runtime/native/src/
	cp python/treelite/common/__init__.py treelite-python/treelite/runtime/native/python/treelite_runtime/common/
	cp python/treelite/common/compat.py treelite-python/treelite/runtime/native/python/treelite_runtime/common/
	cp python/treelite/common/util.py treelite-python/treelite/runtime/native/python/treelite_runtime/common/
	cd treelite-python; python setup.py sdist; mv dist/*.tar.gz ..; cd ..
	rm -rfv treelite-python
