echo ##[section]Running Python tests...
mkdir temp
call activate dev
python -m pytest --basetemp="%WORKING_DIR%\temp" --cov=treelite --cov=treelite_runtime --cov-report xml -v --fulltrace tests\python
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Submitting code coverage data to CodeCov...
python -m codecov -f coverage.xml
if %errorlevel% neq 0 exit /b %errorlevel%
