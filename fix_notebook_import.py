import json

notebook_path = 'd:/tempe/Global_devlopment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# The imports are in the first cell
code_cells = [c for c in data['cells'] if c['cell_type'] == 'code']
target_cell = code_cells[0] 

current_source = target_cell['source']

# Check if already present
if any('AgglomerativeClustering' in line for line in current_source):
    print("AgglomerativeClustering import already exists.")
else:
    # Find the line with KMeans import and add after it
    new_source = []
    for line in current_source:
        new_source.append(line)
        if 'from sklearn.cluster import KMeans' in line:
            new_source.append('from sklearn.cluster import AgglomerativeClustering\\n')
    
    target_cell['source'] = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    
    print(f"✓ Patched {notebook_path} with AgglomerativeClustering import.")
