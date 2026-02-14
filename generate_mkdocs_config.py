import os
import yaml
import re

def clean_name(name):
    # Remove extension
    name = os.path.splitext(name)[0]
    # Remove leading numbers and underscores/hyphens
    name = re.sub(r'^[\d\s\-_]+', '', name)
    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    # Title case
    return name.title()

def generate_nav(dir_path, base_path):
    nav = []
    items = sorted(os.listdir(dir_path))
    
    # Priority for index.md
    if 'index.md' in items:
        nav.append({'Overview': os.path.relpath(os.path.join(dir_path, 'index.md'), base_path).replace('\\', '/')})
        items.remove('index.md')

    for item in items:
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            if item.startswith('.') or item == 'site' or item == '__pycache__' or item.endswith('_files'):
                continue
            sub_nav = generate_nav(item_path, base_path)
            if sub_nav:
                nav.append({clean_name(item): sub_nav})
        elif item.endswith('.md'):
            nav.append({clean_name(item): os.path.relpath(item_path, base_path).replace('\\', '/')})
    return nav

base_dir = r'f:\OpenGit\python'
docs_dir = os.path.join(base_dir, 'docs')

# Generate the navigation
navigation = generate_nav(docs_dir, docs_dir)

# Fix 'Overview' for the very top level (it should be 'Home' or similar)
if navigation and 'Overview' in navigation[0]:
    navigation[0] = {'Home': navigation[0]['Overview']}

config = {
    'site_name': 'Geo Coding Muscle with Python',
    'site_url': 'https://pulakeshpradhan.github.io/python/',
    'theme': {
        'name': 'material',
        'palette': [
            {
                'media': "(prefers-color-scheme: light)",
                'scheme': 'default',
                'primary': 'indigo',
                'accent': 'indigo',
                'toggle': {'icon': 'material/brightness-7', 'name': 'Switch to dark mode'}
            },
            {
                'media': "(prefers-color-scheme: dark)",
                'scheme': 'slate',
                'primary': 'indigo',
                'accent': 'indigo',
                'toggle': {'icon': 'material/brightness-4', 'name': 'Switch to light mode'}
            }
        ],
        'features': [
            'navigation.tabs',
            'navigation.top',
            'search.suggest',
            'search.highlight',
            'content.code.copy'
        ]
    },
    'markdown_extensions': [
        'admonition',
        {'pymdownx.highlight': {'anchor_linenums': True}},
        'pymdownx.inlinehilite',
        'pymdownx.snippets',
        'pymdownx.superfences',
        {'pymdownx.arithmatex': {'generic': True}},
        'attr_list',
        'md_in_html'
    ],
    'nav': navigation
}

with open(os.path.join(base_dir, 'mkdocs.yml'), 'w', encoding='utf-8') as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)

print("mkdocs.yml generated successfully with all markdown files!")
