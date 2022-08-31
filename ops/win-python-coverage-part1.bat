echo "##[section]Setting up Python environment..."
call activate
conda install --yes --quiet -c conda-forge numpy scipy scikit-learn pandas scikit-learn pytest pytest-cov
python -m pip install xgboost lightgbm codecov

echo "##[section]Generating Visual Studio solution..."
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64
