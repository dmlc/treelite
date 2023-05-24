echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64 -DBUILD_CPP_TEST=ON
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Building Visual Studio solution...
cmake --build . --config Release -- /m
if %errorlevel% neq 0 exit /b %errorlevel%
cd ..

echo ##[section]Running C++ tests...
.\build\treelite_cpp_test.exe
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Setting up Python environment...
call micromamba activate dev

echo ##[section]Packaging Python wheel for Treelite...
cd python
pip wheel --no-deps -v . --wheel-dir dist/
if %errorlevel% neq 0 exit /b %errorlevel%
