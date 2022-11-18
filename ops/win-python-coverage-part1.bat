echo ##[section]Setting up Python environment...
conda create -n dev -y -q -c conda-forge python=3.9 numpy scipy scikit-learn pandas scikit-learn pytest pytest-cov xgboost lightgbm || cd .
call activate dev
python -m pip install codecov
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64
if %errorlevel% neq 0 exit /b %errorlevel%
