echo ##[section]Running C++ tests...
.\build\treelite_cpp_test.exe
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Setting up Python environment...
conda create -n dev -y -q -c conda-forge python=3.9 numpy scipy scikit-learn pandas
if %errorlevel% neq 0 exit /b %errorlevel%
call activate dev

echo ##[section]Packaging Python wheel for Treelite...
cd python
python setup.py bdist_wheel --universal
if %errorlevel% neq 0 exit /b %errorlevel%

echo ##[section]Packaging Python wheel for Treelite runtime...
cd ..\runtime\python
python setup.py bdist_wheel --universal
if %errorlevel% neq 0 exit /b %errorlevel%
