import json
from path import Path as path
import pandas as pd
from collections import defaultdict, Counter

def convert_fname(fname):
    output = fname
    output = output.replace(":","")
    output = output.replace("_","__")
    output = output.replace("/","_")
    output = output.replace("\\","_")
    output = str(path(output).stem) + ".jpg"
    return output

def txt_to_list(fname):
    with open(fname,'r', encoding='utf-8') as f:
        return f.read().split("\n")
    
def list_to_txt(mylist, fname):
    with open(fname,'w', encoding='utf-8') as f:
        f.write("\n".join(mylist))
    
def read_json(jsonfile):
    with open(jsonfile, 'r', encoding='utf-8') as f:
        return json.loads(f.read())
    
def write_json(h, jsonfile):
    with open(jsonfile,'w', encoding='utf-8') as f:
        f.write(json.dumps(h))
        
def folder_report(folder, prefix, n):
    counts = defaultdict(int)
    files = [f for f in path(folder).files() if f.name.startswith(prefix)]
    print(len(files))
    for f in files:
        key = f.split("_")[n]
        counts[key] += 1
    keys = list(counts)
    values = [counts[k] for k in keys]
    df = pd.DataFrame()
    df['keys'] = keys
    df['values'] = values
    df.sort_values('values', axis=0, ascending= False, inplace = True)
    return df

def find_and_add_substrings(s, t):
    result = []
    start = 0
    while True:
        index = s.find(t, start)
        if index == -1:
            break
        result.append(s[:index + len(t)])
        start = index + 1
    return result

def get_most_common_folders(file_paths, N):
    folder_counts = Counter()
    
    # Count occurrences of each folder
    for file_path in file_paths:
        for folder in find_and_add_substrings(file_path, "/"):
            folder_counts[folder] += 1
    
    # Sort folders by counts in descending order
    sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N folders
    most_common_folders = sorted_folders[:N]
    
    return most_common_folders