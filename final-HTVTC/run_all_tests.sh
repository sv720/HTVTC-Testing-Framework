#!/bin/bash

# List of Python files to run sequentially
python_files=("KNN-regression-workspace.py" "KNN-classification-workspace.py" "random-forest-workspace.py" "svm-rbf-workspace.py")
# "svm-polynomial-workspace.py"
# Loop through each Python file and run them
for file in "${python_files[@]}"; do
    echo "Running $file..."
    python3 "$file" | grep '^True value:'
done
