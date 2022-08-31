echo "##[section]Running C++ tests..."
.\build\treelite_cpp_test.exe

echo "##[section]Setting up Python environment..."
conda create -n dev -y -q -c conda-forge python=3.9 numpy scipy scikit-learn pandas
call activate dev

echo "##[section]Packaging Python wheel for Treelite..."
cd python
python setup.py bdist_wheel --universal

echo "##[section]Packaging Python wheel for Treelite runtime..."
cd ..\runtime\python
python setup.py bdist_wheel --universal
