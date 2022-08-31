echo ##[section]Running Python tests...
mkdir temp
call activate dev
python -m pytest --basetemp="%WORKING_DIR%\temp" --cov=treelite --cov=treelite_runtime --cov-report xml -v --fulltrace tests\python

echo ##[section]Submitting code coverage data to CodeCov...
python -m codecov -f coverage.xml
