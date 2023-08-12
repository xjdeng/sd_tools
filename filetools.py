import json
from path import Path as path
import pandas as pd
from collections import defaultdict

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