import hashlib
from path import Path as path
import json
import warnings
from tqdm import tqdm

def run(filename):
    try:
        with open(filename, "rb") as f:
            hash_obj = hashlib.sha256()
            while chunk:= f.read(65536):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()[0:10]
    except FileNotFoundError:
        return None
    
def run_dir(mydir, outfile = None):
    result = {}
    files = path(mydir).files()
    for i in tqdm(range(len(files))):
        f = files[i]
        hashval = run(f)
        if hashval in result:
            msg = "Hash {} already found in file {}; make sure they're the same model!".format(hashval, result[hashval])
            warnings.warn(msg)
        result[hashval] = f
    if outfile:
        with open(outfile,'w') as f:
            f.write(json.dumps(result))
    return result