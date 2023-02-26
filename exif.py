from PIL import Image
import pandas as pd
from path import Path as path

def get_user_comment(img):
    if isinstance(img, str):
        img = Image.open(img)
    user_comment = img._getexif().get(37510)
    
    # decode the UserComment into a string
    if user_comment is not None:
        user_comment_string = user_comment.decode("utf-16be")
    else:
        user_comment_string = ""
    return "Prompt: " + user_comment_string[4:]

def decode_comment(input_string):
    result = {}
    neg_i = input_string.find("Negative prompt: ")
    steps_i = input_string.find("Steps: ")
    result['Prompt'] = input_string[len("Prompt: "):neg_i].strip()
    result['Negative prompt'] = input_string[neg_i + len("Negaive prompt: "):steps_i].strip()
    input_str = input_string[steps_i:]
    pairs = input_str.split(", ")
    for pair in pairs:
        # Split the key and value on the colon character
        key, value = pair.split(": ")
        # Strip any leading or trailing whitespace from the key and value
        key = key.strip()
        value = value.strip()
        # Convert the value to an appropriate data type (integer, float, etc.)
        if value.isnumeric():
            value = int(value)
        elif "." in value and all(part.isnumeric() for part in value.split(".")):
            value = float(value)
        # Add the key-value pair to the dictionary
        result[key] = value    
    return result

def run(img):
    return decode_comment(get_user_comment(img))

def run_report(filelist, keys= ['Steps', 'CFG scale', 'Size', 'Model hash', 'Hires upscale'], output = None):
    report = pd.DataFrame(index = filelist)
    key_results = {k:[] for k in keys}
    for f in filelist:
        for k in keys:
            key_results[k].append(run(f).get(k))
    for k in keys:
        report[k] = key_results[k]
    if output:
        report.to_csv(output)
    return report

def run_dir(dirname, *args, **kwargs):
    return run_report(path(dirname).files(), *args, **kwargs)