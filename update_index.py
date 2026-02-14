import os

def generate_text_tree(dir_path, prefix=""):
    lines = []
    items = sorted(os.listdir(dir_path))
    
    # Filter items: hide system files, _files folders, datasets (csv, xlsx), and images
    exclude_dirs = {'.github', '.git', 'site', '__pycache__', '.ipynb_checkpoints'}
    exclude_exts = {'.csv', '.xlsx', '.png', '.jpg', '.jpeg', '.ico', '.zip', '.dbf', '.shp', '.shx', '.prj', '.cpg', '.geojson', '.pkl'}
    
    # Actually, the user might want some files but not others. 
    # Let's keep .md, .py, .js, .ipynb and directories.
    
    filtered_items = []
    for item in items:
        if item.startswith('.') or item in exclude_dirs:
            continue
        if item.endswith('_files'):
            continue
        if item == 'index.md':
            continue
            
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext not in {'.md', '.py', '.js', '.ipynb', '.pdf'}:
                continue
        
        filtered_items.append(item)
    
    for i, item in enumerate(filtered_items):
        is_last = (i == len(filtered_items) - 1)
        connector = "└── " if is_last else "├── "
        
        item_path = os.path.join(dir_path, item)
        clean_name = item.replace('_', ' ').replace('-', ' ').title()
        import re
        clean_name = re.sub(r'^\d+\s+', '', clean_name)
        
        # Add icon based on type
        icon = "📁" if os.path.isdir(item_path) else "📄"
        if item.endswith('.pdf'): icon = "📕"
        if item.endswith('.ipynb'): icon = "📓"
        
        lines.append(f"{prefix}{connector}{icon} {clean_name}")
        
        if os.path.isdir(item_path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(generate_text_tree(item_path, new_prefix))
            
    return lines

docs_dir = r'f:\OpenGit\python\docs'
content = [
    "# Geo Coding Muscle with Python",
    "",
    "Welcome to the comprehensive repository of Python learning resources, projects, and geospatial analysis guides.",
    "",
    "## 📚 Repository Structure",
    "",
    "Explore the various sections of this documentation through the interactive tree below. Note that for most tutorials, an **Open in Colab** button is available at the top of the page.",
    "",
    "```text",
]

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

print("index.md updated with refined tree structure!")
