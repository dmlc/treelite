echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 17 2022" -A x64 -DBUILD_CPP_TEST=ON
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Building Visual Studio solution...
cmake --build . --config Release -- /m
if %errorlevel% neq 0 exit /b %errorlevel%
cd ..

echo ##[section]Running C++ tests...
.\build\treelite_cpp_test.exe
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Packaging Python wheel for Treelite...
cd python
pip wheel --no-deps -v . --wheel-dir dist/
if %errorlevel% neq 0 exit /b %errorlevel%
