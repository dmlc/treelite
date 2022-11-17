echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64 -DBUILD_CPP_TEST=ON
if %errorlevel% neq 0 exit /b %errorlevel%
