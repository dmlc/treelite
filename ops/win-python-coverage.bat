echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Building Visual Studio solution...
cmake --build . --config Release -- /m
if %errorlevel% neq 0 exit /b %errorlevel%
cd ..

echo ##[section]Running Python tests...
mkdir temp
set "PYTHONPATH=./python"
set "PYTEST_TMPDIR=%USERPROFILE%\AppData\Local\Temp\pytest_temp"
mkdir "%PYTEST_TMPDIR%"
python -m pytest --basetemp="%USERPROFILE%\AppData\Local\Temp\pytest_temp" --cov=treelite --cov-report xml -v -rxXs --fulltrace --durations=0 tests\python
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Submitting code coverage data to CodeCov...
python -m codecov -f coverage.xml
if %errorlevel% neq 0 exit /b %errorlevel%
