import cv2
import numpy as np
from path import Path as path
import itertools, random, math

def center_crop(image, a):
    if isinstance(image, str):
        image = cv2.imread(image)
    # Read the image using OpenCV

    # Calculate the dimensions for center cropping
    height, width, _ = image.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    end_x = start_x + min_dim
    end_y = start_y + min_dim

    # Perform the center crop
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Resize the cropped image to a x a
    resized_image = cv2.resize(cropped_image, (a, a))

    # Convert the image to a NumPy array
    numpy_image = np.array(resized_image)

    return numpy_image

def resize(img, dim = 512):
    if isinstance(img, str):
        img = cv2.imread(img)
    try:
        h,w = img.shape[0:2]
    except Exception:
        return None
    newimg = 255*np.ones((dim, dim, 3))
    new_w = int(round(w*dim/h))
    if new_w > dim:
        new_h = int(round(h*dim/w))
        img2 = cv2.resize(img, (dim, new_h))
        offset = int(round((dim - new_h)/2))
        newimg[offset:offset+new_h,:,:] = img2
    else:
        img2 = cv2.resize(img, (new_w, dim))
        offset = int(round((dim - new_w)/2))
        newimg[:,offset:offset+new_w,:] = img2
    return newimg

def hash_img(image, hashSize=8):
    #float_image = image.astype(np.float64)
    #converted_image = cv2.convertScaleAbs(float_image)
    #image = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
    if isinstance(image, str):
        image = cv2.imread(image)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        float_image = image.astype(np.float64)
        converted_image = cv2.convertScaleAbs(float_image)
        image = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)        
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def resize_run(srcdir, fnoffset = 0, label = "", dim = 512, destdir = None):
    for i,f in enumerate(path(srcdir).files()):
        if not destdir:
            destdir = "{}/512".format(srcdir)
        path(destdir).mkdir_p()
        img = cv2.imread(f)
        try:
            h,w = img.shape[0:2]
        except Exception as e:
            print(f, e)
            continue
        newimg = 255*np.ones((dim, dim, 3))
        new_w = int(round(w*dim/h))
        if new_w > dim:
            new_h = int(round(h*dim/w))
            img2 = cv2.resize(img, (dim, new_h))
            offset = int(round((dim - new_h)/2))
            newimg[offset:offset+new_h,:,:] = img2
        else:
            img2 = cv2.resize(img, (new_w, dim))
            offset = int(round((dim - new_w)/2))
            newimg[:,offset:offset+new_w,:] = img2
        j = i + fnoffset
        cv2.imwrite("{}/{}{}".format(destdir, j, f.ext), newimg)
        with open("{}/{}.txt".format(destdir, j),'w') as ff:
            ff.write(label)
            
def resize_center(srcdir, fnoffset = 0, label = ""):
    for i,f in enumerate(path(srcdir).files()):
        destdir = "{}/512".format(srcdir)
        path(destdir).mkdir_p()
        img = cv2.imread(f)
        try:
            h,w = img.shape[0:2]
        except Exception as e:
            print(f, e)
            continue
        newimg = center_crop(img, 512)
        try:
            j = i + fnoffset
        except TypeError:
            j = 0
        if fnoffset is None:
            fname = f.stem
        else:
            fname = j
        cv2.imwrite("{}/{}{}".format(destdir, fname, f.ext), newimg)
        with open("{}/{}.txt".format(destdir, fname),'w') as ff:
            ff.write(label)        
            
def replace_substrings(text, replacements):
    replaced_text = text
    visited_indices = set()
    
    for pattern, replacement in replacements:
        start = 0
        while True:
            index = replaced_text.find(pattern, start)
            if index == -1:
                break
            if any(start <= i < index + len(pattern) for i, _ in visited_indices):
                # Ignore overlapping matches
                start = index + len(pattern)
                continue

            replaced_text = replaced_text[:index] + replacement + replaced_text[index+len(pattern):]
            visited_indices.add((index, index+len(pattern)))
            start = index + len(replacement)

    return replaced_text