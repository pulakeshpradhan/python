import os
import yaml

with open(r'f:\OpenGit\python\mkdocs.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def get_nav_files(nav_item):
    files = []
    if isinstance(nav_item, str):
        files.append(nav_item)
    elif isinstance(nav_item, dict):
        for val in nav_item.values():
            files.extend(get_nav_files(val))
    elif isinstance(nav_item, list):
        for item in nav_item:
            files.extend(get_nav_files(item))
    return files

nav_files = set(get_nav_files(config.get('nav', [])))

all_md_files = []
docs_dir = r'f:\OpenGit\python\docs'
for root, dirs, files in os.walk(docs_dir):
    for file in files:
        if file.endswith('.md'):
            rel_path = os.path.relpath(os.path.join(root, file), docs_dir).replace('\\', '/')
            all_md_files.append(rel_path)

missing = [f for f in all_md_files if f not in nav_files]
print(f"Missing files: {missing}")
