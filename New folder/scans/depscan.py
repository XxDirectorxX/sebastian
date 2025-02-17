import os
import csv
import ast

def find_imports(file_path):
    """Scan a Python file for all import statements."""
    imports = set()
    try:
        with open(file_path, 'r') as file:
            tree = ast.parse(file.read(), filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return imports

def scan_directory_for_dependencies(directory):
    """Scan the directory and all Python files for imports."""
    dependencies = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports = find_imports(file_path)
                dependencies.update(imports)
    return dependencies

def save_dependencies_to_csv(dependencies, output_file):
    """Save the list of dependencies to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Library/Dependency'])
        for dep in sorted(dependencies):
            writer.writerow([dep])

# Set the path to your directory and output file
directory_to_scan = r'R:\sebastian'
output_csv_file = r'R:\sebastian\dep-lib-scan.csv'

# Scan the directory and save the dependencies to CSV
dependencies = scan_directory_for_dependencies(directory_to_scan)
save_dependencies_to_csv(dependencies, output_csv_file)

print(f"Dependencies have been saved to {output_csv_file}")
