import os
import ast
import sys
import importlib.metadata
from collections import defaultdict
from argparse import ArgumentParser


# Common mappings for package names that differ from import names
COMMON_PACKAGE_MAPPINGS = {
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "tkinter": "tk",
    # Add more as needed
}

def find_imported_packages(directory):
    """Walk through all .py files in the directory and extract root-level package names."""
    package_set = set()
    built_in_modules = set(sys.builtin_module_names)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError:
                        print(f"Skipping file with syntax error: {file_path}")
                        continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.name.split('.')[0]
                            if name not in built_in_modules and not name.startswith('_'):
                                package_set.add(name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and not node.module.startswith('.'):
                            name = node.module.split('.')[0]
                            if name not in built_in_modules and not name.startswith('_'):
                                package_set.add(name)
    return package_set


def resolve_package_versions(packages):
    """Try to resolve package names to their installed versions."""
    package_versions = {}

    for package in packages:
        # Try mapping first
        resolved_name = COMMON_PACKAGE_MAPPINGS.get(package, package)

        try:
            version = importlib.metadata.version(resolved_name)
            package_versions[resolved_name] = version
        except importlib.metadata.PackageNotFoundError:
            # Fallback to original name if mapped
            if resolved_name != package:
                try:
                    version = importlib.metadata.version(package)
                    package_versions[package] = version
                    continue
                except importlib.metadata.PackageNotFoundError:
                    pass
            print(f"Package '{package}' not found. Skipping.")

    return package_versions


def write_requirements(package_versions, output_file):
    """Write the final requirements.txt file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for package, version in sorted(package_versions.items()):
            f.write(f"{package}=={version}\n")


def main():
    parser = ArgumentParser(description="Generate requirements.txt based on imports used in Python files.")
    parser.add_argument("directory", nargs="?", default=".", help="Root directory of the Python project (default: current directory)")
    parser.add_argument("-o", "--output", default="requirements.txt", help="Output file path (default: requirements.txt)")

    args = parser.parse_args()

    print(f"Scanning Python files in '{args.directory}'...")
    packages = find_imported_packages(args.directory)

    print(f"Found {len(packages)} unique imported packages.")
    print("Resolving versions...")

    package_versions = resolve_package_versions(packages)

    print(f"Resolved {len(package_versions)} packages.")
    write_requirements(package_versions, args.output)

    print(f"Requirements written to '{args.output}'.")


if __name__ == "__main__":
    main()