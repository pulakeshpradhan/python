import os
import yaml
import re

def generate_nav(dir_path, base_path):
    nav = []
    items = sorted(os.listdir(dir_path))
    
    # Priority for index.md
    if 'index.md' in items:
        nav.append({'Home' if dir_path == base_path else 'Overview': os.path.relpath(os.path.join(dir_path, 'index.md'), base_path).replace('\\', '/')})
        items.remove('index.md')

    for item in items:
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            if item.startswith('.') or item == 'site' or item == '__pycache__':
                continue
            sub_nav = generate_nav(item_path, base_path)
            if sub_nav:
                # Clean up the name (remove numbers, underscores)
                name = item.replace('_', ' ').replace('-', ' ').title()
                # Remove leading numbers if present (e.g. "01 Introduction" -> "Introduction")
                name = re.sub(r'^\d+\s+', '', name)
                nav.append({name: sub_nav})
        elif item.endswith('.md'):
            name = os.path.splitext(item)[0].replace('_', ' ').replace('-', ' ').title()
            name = re.sub(r'^\d+\s+', '', name)
            nav.append({name: os.path.relpath(item_path, base_path).replace('\\', '/')})
    return nav

base_dir = r'f:\OpenGit\python'
docs_dir = os.path.join(base_dir, 'docs')
navigation = generate_nav(docs_dir, docs_dir)

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

print("mkdocs.yml generated successfully!")
