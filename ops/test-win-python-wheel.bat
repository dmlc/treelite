echo ##[section]Setting up Python environment...
conda install -c conda-forge -y mamba
mamba env create -q -f ops/conda_env/dev.yml
call activate dev

echo ##[section]Installing Treelite into Python environment...
for /R %%i in (main\\*.whl) DO python -m pip install "%%i"
for /R %%i in (runtime\\*.whl) DO python -m pip install "%%i"

echo ##[section]Running Python tests...
mkdir temp
python -m pytest --basetemp="%WORKING_DIR%\temp" -v -rxXs --fulltrace tests\python\test_basic.py

echo ##[section]Uploading Python wheels...
for /R %%i in (main\\*.whl) DO python tests\ci_build\rename_whl.py "%%i" %COMMIT_ID% win_amd64
for /R %%i in (runtime\\*.whl) DO python tests\ci_build\rename_whl.py "%%i" %COMMIT_ID% win_amd64
for /R %%i in (main\\*.whl) DO python -m awscli s3 cp "%%i" s3://treelite-wheels/ --acl public-read || cd .
for /R %%i in (runtime\\*.whl) DO python -m awscli s3 cp "%%i" s3://treelite-wheels/ --acl public-read || cd .
