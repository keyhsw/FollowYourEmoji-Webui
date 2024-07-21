import os
from pathlib import Path

def tree(directory):
    print(f'📁 {os.path.basename(directory)}')
    directory = Path(directory)
    print_tree(directory, level=0)

def print_tree(directory, level):
    padding = '    ' * level
    for path in sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if path.name.startswith('.'):
            continue
        if path.is_dir():
            print(f'{padding}├── 📁 {path.name}')
            print_tree(path, level + 1)
        else:
            print(f'{padding}├── 📄 {path.name}')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Использование: python script.py <путь<em>к</em>директории>")
    else:
        tree(sys.argv[1])
