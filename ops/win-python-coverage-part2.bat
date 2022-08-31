echo "Running Python tests..."
call activate
mkdir temp
python -m pytest --basetemp="%WORKING_DIR%\temp" --cov=treelite --cov=treelite_runtime --cov-report xml -v --fulltrace tests\python

echo "Submitting code coverage data to CodeCov..."
python -m codecov -f coverage.xml
