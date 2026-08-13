import re

with open("build_notebooks.py", "r", encoding="utf-8") as f:
    content = f.read()

# The python code we injected had unescaped triple quotes which broke the f-string.
# We will use single quotes for the docstring inside the injected python code.
content = content.replace('"""Scan audio directory and build a manifest DataFrame, loading references from CSV if available."""', "'''Scan audio directory and build a manifest DataFrame, loading references from CSV if available.'''")

with open("build_notebooks.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch fixed successfully.")
