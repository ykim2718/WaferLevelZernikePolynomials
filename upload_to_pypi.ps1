$python = "c:/Y/anaconda3/python.exe "
& $python -m pip install --upgrade build twine
& $python -m build
& $python -m twine upload dist/*
