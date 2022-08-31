echo "##[section]Running C++ tests..."
.\build\treelite_cpp_test.exe

echo "##[section]Setting up Python environment..."
call activate
conda install --yes --quiet -c conda-forge numpy scipy scikit-learn pandas

echo "##[section]Packaging Python wheel for Treelite..."
cd python
python setup.py bdist_wheel --universal

echo "##[section]Packaging Python wheel for Treelite runtime..."
cd ..\runtime\python
python setup.py bdist_wheel --universal
