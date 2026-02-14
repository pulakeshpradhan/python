import os

def add_colab_buttons(docs_dir, repo_url_base):
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md') and file != 'index.md':
                md_path = os.path.join(root, file)
                ipynb_file = file.replace('.md', '.ipynb')
                ipynb_path = os.path.join(root, ipynb_file)
                
                # Check if corresponding .ipynb exists
                if os.path.exists(ipynb_path):
                    # Calculate relative path from docs root for the URL
                    rel_path = os.path.relpath(ipynb_path, docs_dir).replace('\\', '/')
                    colab_url = f"https://colab.research.google.com/github/pulakeshpradhan/python/blob/main/docs/{rel_path}"
                    badge_md = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n\n"
                    
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Avoid duplicate badges
                    if "colab-badge.svg" not in content:
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(badge_md + content)
                        print(f"Added Colab button to: {md_path}")

docs_dir = r'f:\OpenGit\python\docs'
repo_url_base = "https://github.com/pulakeshpradhan/python"
add_colab_buttons(docs_dir, repo_url_base)
print("Finished adding Colab buttons.")
