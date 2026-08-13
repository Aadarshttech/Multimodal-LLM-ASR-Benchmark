import re

with open("build_notebooks.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the processor kwarg from 'audios=audios' to 'audio=audios'
content = content.replace("audios=audios", "audio=audios")

with open("build_notebooks.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied: audios -> audio")
