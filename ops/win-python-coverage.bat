echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Building Visual Studio solution...
cmake --build . --config Release -- /m
if %errorlevel% neq 0 exit /b %errorlevel%
cd ..

echo ##[section]Running Python tests...
mkdir temp
call micromamba activate dev
if %errorlevel% neq 0 exit /b %errorlevel%
set "PYTHONPATH=./python;./runtime/python"
set "PYTEST_TMPDIR=%WORKING_DIR%\temp"
python -m pytest --basetemp="%WORKING_DIR%\temp" --cov=treelite --cov=treelite_runtime --cov-report xml -v -rxXs --fulltrace --durations=0 tests\python
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Submitting code coverage data to CodeCov...
python -m codecov -f coverage.xml
if %errorlevel% neq 0 exit /b %errorlevel%
