import os

def generate_markdown_tree(dir_path, level=0):
    tree_lines = []
    items = sorted(os.listdir(dir_path))
    
    # Filter items
    items = [i for i in items if not i.startswith('.') and i != 'index.md' and i != 'site' and i != '__pycache__']
    
    for item in items:
        item_path = os.path.join(dir_path, item)
        indent = "    " * level
        
        # Clean up name
        clean_name = item.replace('_', ' ').replace('-', ' ').title()
        import re
        clean_name = re.sub(r'^\d+\s+', '', clean_name)
        
        if os.path.isdir(item_path):
            tree_lines.append(f"{indent}- **{clean_name}**")
            tree_lines.extend(generate_markdown_tree(item_path, level + 1))
        elif item.endswith('.md'):
            # Only include markdown files
            name = os.path.splitext(clean_name)[0]
            rel_link = os.path.relpath(item_path, os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
            # Rel link from docs/index.md
            rel_link_from_index = os.path.relpath(item_path, r'f:\OpenGit\python\docs').replace('\\', '/')
            tree_lines.append(f"{indent}- [{name}]({rel_link_from_index})")
            
    return tree_lines

docs_dir = r'f:\OpenGit\python\docs'
tree = generate_markdown_tree(docs_dir)

content = [
    "# Geo Coding Muscle with Python",
    "",
    "Welcome to the comprehensive repository of Python learning resources, projects, and geospatial analysis guides.",
    "",
    "## 📚 Repository Structure",
    "",
    "Explore the various sections of this documentation through the interactive tree below:",
    "",
    "```text",
]

# Simple text tree for professional look
def generate_text_tree(dir_path, prefix=""):
    lines = []
    items = sorted(os.listdir(dir_path))
    items = [i for i in items if not i.startswith('.') and i != 'index.md' and i != 'site' and i != '__pycache__']
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        item_path = os.path.join(dir_path, item)
        clean_name = item.replace('_', ' ').replace('-', ' ').title()
        import re
        clean_name = re.sub(r'^\d+\s+', '', clean_name)
        
        lines.append(f"{prefix}{connector}{clean_name}")
        
        if os.path.isdir(item_path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(generate_text_tree(item_path, new_prefix))
            
    return lines

content.extend(generate_text_tree(docs_dir))
content.append("```")
content.append("")
content.append("## 🚀 Getting Started")
content.append("Use the navigation tabs above to dive into specific domains:")
content.append("- **Fundamentals**: Core Python concepts and basics.")
content.append("- **Data Science**: Pandas, NumPy, and data manipulation.")
content.append("- **Deep Learning**: PyTorch and neural network projects.")
content.append("- **Geospatial**: Remote sensing, GEE, and spatial analysis.")
content.append("- **Projects**: Real-world applications and forecasting models.")

with open(os.path.join(docs_dir, 'index.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(content))

print("index.md updated with professional tree structure!")
