echo ##[section]Setting up Python environment...
conda install -c conda-forge -q -y mamba>=1.0.0
if %errorlevel% neq 0 exit /b %errorlevel%
mamba env create -q -f ops/conda_env/dev.yml
if %errorlevel% neq 0 exit /b %errorlevel%
call activate dev

echo ##[section]Installing Treelite into Python environment...
setlocal enabledelayedexpansion
for /R %%i in (main\\*.whl) DO (
  python tests\ci_build\rename_whl.py "%%i" %COMMIT_ID% win_amd64
  if !errorlevel! neq 0 exit /b !errorlevel!
)
for /R %%i in (runtime\\*.whl) DO (
  python tests\ci_build\rename_whl.py "%%i" %COMMIT_ID% win_amd64
  if !errorlevel! neq 0 exit /b !errorlevel!
)
for /R %%i in (main\\*.whl) DO (
  python -m pip install "%%i"
  if !errorlevel! neq 0 exit /b !errorlevel!
)
for /R %%i in (runtime\\*.whl) DO (
  python -m pip install "%%i"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

echo ##[section]Running Python tests...
mkdir temp
python -m pytest --basetemp="%WORKING_DIR%\temp" -v -rxXs --fulltrace tests\python\test_basic.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Uploading Python wheels...
for /R %%i in (main\\*.whl) DO (
  python -m awscli s3 cp "%%i" s3://treelite-wheels/ --acl public-read || cd .
)
for /R %%i in (runtime\\*.whl) DO (
  python -m awscli s3 cp "%%i" s3://treelite-wheels/ --acl public-read || cd .
)
