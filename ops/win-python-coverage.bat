echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Building Visual Studio solution...
cmake --build . --config Release -- /m
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Running Python tests...
mkdir temp
python -m pytest --basetemp="%WORKING_DIR%\temp" --cov=treelite --cov=treelite_runtime --cov-report xml -v -rxXs --fulltrace tests\python
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Submitting code coverage data to CodeCov...
python -m codecov -f coverage.xml
if %errorlevel% neq 0 exit /b %errorlevel%
