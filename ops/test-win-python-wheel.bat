echo ##[section]Installing Treelite into Python environment...
setlocal enabledelayedexpansion
python tests\ci_build\rename_whl.py python\dist %COMMIT_ID% win_amd64
if %errorlevel% neq 0 exit /b %errorlevel%
for /R %%i in (python\\dist\\*.whl) DO (
  python -m pip install --force-reinstall "%%i"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

echo ##[section]Running Python tests...
mkdir temp
python -m pytest --basetemp="%WORKING_DIR%\temp" -v -rxXs --fulltrace --durations=0 tests\python\test_sklearn_integration.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Uploading Python wheels...
for /R %%i in (python\\dist\\*.whl) DO (
  python -m awscli s3 cp "%%i" s3://treelite-wheels/ --acl public-read --region us-west-2 || cd .
)
