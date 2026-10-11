import os
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    def replacer(match):
        text = match.group(0)
        # Find all `prob[ ... ]` and change them to `prob[ ... ][0]`.
        # Exclude if it already has `[0]`
        text = re.sub(r'(prob\[[^\]]+\])(?!\[)', r'\1[0]', text)
        return text

    new_content = re.sub(r'summary_data\s*=\s*\([^\)]+\)', replacer, content, flags=re.DOTALL)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")

for root, dirs, files in os.walk('example_cycles'):
    for file in files:
        if file.endswith('.py') and 'tests' not in root:
            filepath = os.path.join(root, file)
            fix_file(filepath)
