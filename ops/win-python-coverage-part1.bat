echo ##[section]Setting up Python environment...
conda install -c conda-forge -y mamba
mamba env create -q -f ops/conda_env/dev.yml || cd .
call activate dev
python -m pip install codecov
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64
if %errorlevel% neq 0 exit /b %errorlevel%
