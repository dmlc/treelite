echo ##[section]Setting up Python environment...
conda create -n dev -y -q -c conda-forge python=3.9 numpy scipy scikit-learn pandas scikit-learn pytest pytest-cov || cd .
call activate dev
python -m pip install xgboost lightgbm codecov

echo ##[section]Generating Visual Studio solution...
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64
