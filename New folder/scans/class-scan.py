import os
import csv

# Function to scan directory for all Python (.py) and Quantum Processor (.qp) classes
def scan_for_classes(directory):
    # List to hold found classes
    class_definitions = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".qp")):  # Consider Python and Quantum Processor files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                        # Scan through each line for class definitions
                        for line in lines:
                            if line.strip().startswith("class "):  # Look for class definitions
                                class_name = line.strip().split("(")[0].split()[1]  # Get class name
                                class_definitions.append({
                                    "file": file_path,
                                    "class_name": class_name
                                })
                                print(f"Found class: {class_name} in {file_path}")  # Debug print
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return class_definitions


# Function to save the found classes to a CSV file
def save_classes_to_csv(classes, output_file):
    if classes:
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['class_name', 'file_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for class_info in classes:
                writer.writerow({
                    'class_name': class_info['class_name'],
                    'file_path': class_info['file']
                })
        print(f"Classes saved to {output_file}")
    else:
        print("No classes found to save.")


# Main function
def main():
    directory = "R:/sebastian"  # Specify the directory path
    classes = scan_for_classes(directory)  # Get all class definitions
    output_file = "class.csv"  # Specify the CSV output file
    save_classes_to_csv(classes, output_file)  # Save classes to CSV


if __name__ == "__main__":
    main()
