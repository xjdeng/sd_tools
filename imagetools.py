import cv2
import numpy as np
from path import Path as path
import itertools, random, math
import pandas as pd
import ast
import os
import pyzipper
import zipfile
import tempfile
import sys
import random
import piexif
from PIL import Image, PngImagePlugin


def get_dims512():
    DIMS512 = [(256, 832), (256, 896), (256, 960), (256, 1024), (320, 704), (320, 768), (448, 576), (384, 640)]
    tmp = [(x[1],x[0]) for x in DIMS512]
    DIMS512 += tmp
    DIMS512.append((512,512))
    return DIMS512

def get_dims1024():
    return [(1024,1024),(1152,896),(896,1152),(1216,832),(832,1216),(1344,768),(768,1344),(1536,640),(640,1536)]

def load_image(im_path):
    img = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Convert to 3 channels if the image has 4 channels
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def rotate_image_90(file_path, left = True):
    if left:
        direction = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        direction = cv2.ROTATE_90_CLOCKWISE
    try:
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Unable to read the file '{file_path}'. Make sure the file exists and is an image.")
            return

        # Rotate the image 90 degrees clockwise
        rotated_image = cv2.rotate(image, direction)

        # Overwrite the original file
        cv2.imwrite(file_path, rotated_image)
        print(f"Image rotated 90 degrees and saved as '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def detect_white_borders(img_path, tolerance=3):
    """
    Detects if an image contains white borders.
    
    Parameters:
    img_path (str): Path to the image.
    tolerance (int): Minimum size of the border to detect (in pixels).
    
    Returns:
    bool: True if white borders are detected, False otherwise.
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    
    # Check image dimensions
    h, w, _ = img.shape
    
    # Define a white threshold
    white_threshold = 250  # Slightly less than 255 to account for slight variations
    
    # Check top border
    top_border = np.all(img[:tolerance, :, :] > white_threshold, axis=-1)
    if np.all(top_border):
        return True
    
    # Check bottom border
    bottom_border = np.all(img[-tolerance:, :, :] > white_threshold, axis=-1)
    if np.all(bottom_border):
        return True
    
    # Check left border
    left_border = np.all(img[:, :tolerance, :] > white_threshold, axis=-1)
    if np.all(left_border):
        return True
    
    # Check right border
    right_border = np.all(img[:, -tolerance:, :] > white_threshold, axis=-1)
    if np.all(right_border):
        return True
    
    return False

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

def resize_image_to_height(image, target_height):
    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Get the original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate the new width to preserve the aspect ratio
    aspect_ratio = original_width / original_height
    new_width = int(target_height * aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, target_height))
    
    return resized_image


def quick_thumbnail(imgpath = "./", outfile = "thumbnail.jpg", images = 3, dims = (1280, 720)):
    files = [f for f in path(imgpath).files() if f.abspath() != path(outfile).abspath()]
    random.shuffle(files)
    queue = []
    height = None
    while True:
        if (len(queue) == images) or (not files):
            break
        test = files.pop()
        img = cv2.imread(test)
        if img is not None:
            if height is None:
                height = img.shape[0]
            if img.shape[0] != height:
                img = resize_image_to_height(img, height)
            queue.append(img)
    width = sum([img.shape[1] for img in queue])
    newimg = np.zeros((height, width, 3))
    offset = 0
    for img in queue:
        newimg[:,offset:offset + img.shape[1],:] = img
        offset += img.shape[1]
    resized = cv2.resize(newimg, dims)
    cv2.imwrite(outfile, resized)


def resize_bucket_old(img, buckets = [(1024, 512), (1024, 768), (1024, 1024), (512, 1024), (768, 1024)]):
    if isinstance(img, str):
        img = load_image(img)
    try:
        h, w = img.shape[0:2]
    except Exception:
        return None
    
    aspect_ratio = w / h
    best_bucket = None
    min_diff = float('inf')
    
    # Find the best bucket based on aspect ratio
    for bucket in buckets:
        bucket_w, bucket_h = bucket
        bucket_aspect_ratio = bucket_w / bucket_h
        diff = abs(aspect_ratio - bucket_aspect_ratio)
        if diff < min_diff:
            min_diff = diff
            best_bucket = bucket
    
    if best_bucket is None:
        return None
    bucket_w, bucket_h = best_bucket
    newimg = 255 * np.ones((bucket_h, bucket_w, 3), dtype=np.uint8)
    
    # Resize image while maintaining aspect ratio
    if aspect_ratio > (bucket_w / bucket_h):
        new_w = bucket_w
        new_h = int(round(h * bucket_w / w))
    else:
        new_h = bucket_h
        new_w = int(round(w * bucket_h / h))
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Calculate offsets to center the image
    offset_y = (bucket_h - new_h) // 2
    offset_x = (bucket_w - new_w) // 2
    
    newimg[offset_y:offset_y + new_h, offset_x:offset_x + new_w, :] = img_resized
    return newimg

def resize_bucket(img, buckets = None):
    return resize_bucket_crop(img, buckets)

def resize_bucket_crop(img, buckets=None):
    if not buckets:
        buckets = get_dims512()
    if isinstance(img, str):
        img = load_image(img)
    
    try:
        h, w = img.shape[0:2]
    except Exception:
        return None
    
    aspect_ratio = w / h
    best_bucket = None
    min_diff = float('inf')
    
    # Find the best bucket based on aspect ratio
    for bucket in buckets:
        bucket_w, bucket_h = bucket
        bucket_aspect_ratio = bucket_w / bucket_h
        diff = abs(aspect_ratio - bucket_aspect_ratio)
        if diff < min_diff:
            min_diff = diff
            best_bucket = bucket
    
    if best_bucket is None:
        return None
    
    bucket_w, bucket_h = best_bucket
    bucket_aspect_ratio = bucket_w / bucket_h
    
    # Crop image while maintaining the closest aspect ratio
    if aspect_ratio > bucket_aspect_ratio:
        # Image is too wide, crop width
        new_w = int(h * bucket_aspect_ratio)
        new_h = h
    else:
        # Image is too tall, crop height
        new_h = int(w / bucket_aspect_ratio)
        new_w = w
    
    # Compute cropping coordinates
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped_img = img[start_y:start_y + new_h, start_x:start_x + new_w]
    
    # Resize cropped image to exactly match the bucket dimensions
    resized_img = cv2.resize(cropped_img, (bucket_w, bucket_h))
    
    return resized_img


def resize_run_bucket(srcdir, fnoffset = 0, label = "", buckets = None, destdir = None):
    if not buckets:
        buckets = get_dims512()
    for i,f in enumerate(path(srcdir).files()):
        if not destdir:
            destdir = "{}/512".format(srcdir)
        path(destdir).mkdir_p()
        newimg = resize_bucket(f, buckets)
        if newimg is None:
            print(f"Warning: no image found in {f}")
            continue
        j = i + fnoffset
        cv2.imwrite("{}/{}.jpg".format(destdir, j), newimg)
        with open("{}/{}.txt".format(destdir, j),'w') as ff:
            ff.write(label)    
    

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

def add_text_to_image(img, text, textheight = 0.01, text_color=(0, 0, 0), border_color=(255, 255, 255), \
                      border_size = 2):
    # Read the image using OpenCV
    if isinstance(img, str):
        img = load_image(img)
    if img is None:
        return None
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate margins
    left_margin = int(width * 0.01)
    bottom_margin = int(height * 0.01)
    
    # Calculate maximum height of text
    max_text_height = int(textheight * height)
    
    # Choose font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    
    # Adjust font size to fit maximum height
    text_width, text_height = cv2.getTextSize(text, font, scale, thickness=1)[0]
    while text_height < max_text_height:
        scale += 1
        text_width, text_height = cv2.getTextSize(text, font, scale, thickness=1)[0]
    
    # Calculate text position
    text_position = (left_margin, height - bottom_margin)
    
    # Draw text with border
    
    img = cv2.putText(img, text, text_position, font, scale, border_color, thickness=border_size, lineType=cv2.LINE_AA)
    img = cv2.putText(img, text, text_position, font, scale, text_color, thickness=1, lineType=cv2.LINE_AA)
    return img

def add_text_dir(dir, *args, **kwargs):
    for f in path(dir).files():
        img = add_text_to_image(f, *args, **kwargs)
        if img is None:
            continue
        #cv2.imwrite(f, img)
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        im_buf_arr.tofile(f)

def to_gpt_prompt(listoflists, header = None, forbidden = None):
    if not header:
        text = "Generate more examples like the following:\n\n"
    else:
        text = header.strip() + "\n\n"
    if not forbidden:
        forbidden = ["good aesthetic", "good quality", "good contrast", "good blur", "good noise", \
                     "poor aesthetic", "poor quality", "poor contrast", "poor blur", "poor noise", \
                     "portrait dimensions", "landscape dimensions"]
    forbidden = set(forbidden)
    for i,tags in enumerate(listoflists):
        tags = [t for t in tags if t not in forbidden]
        example = ", ".join(tags)
        text += f"{i+1}: {example}\n\n"
    text += f"{len(listoflists) + 1}:"
    return text

def gpt_from_folder(folder, tagcsv, outfile = "output.txt", **kwargs):
    tagdf = pd.read_csv(tagcsv)
    tagdict = {row['file']:ast.literal_eval(row['tags']) for _,row in tagdf.iterrows()}
    lofl = []
    for f in path(folder).files():
        tags = tagdict.get(str(f.name))
        if not tags:
            continue
        else:
            lofl.append(tags)
    prompt = to_gpt_prompt(lofl, **kwargs)
    with open(outfile,'w') as f:
        f.write(prompt)

def calculate_mse(imageA, imageB):
    # Compute the Mean Squared Error between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def find_labels(img, mydir):
    if isinstance(img, str):
        img = cv2.imread(img)
    h0,w0 = img.shape[0:2]
    imgs = []
    errors = []
    files = path(mydir).files()
    for f in files:
        pic = cv2.imread(f)
        imgs.append(pic)
        if pic is None:
            errors.append(float('inf'))
        else:
            h,w = pic.shape[0:2]
            if (h != h0) or (w != w0):
                img2 = cv2.resize(img, (w, h))
            else:
                img2 = img
            error = calculate_mse(img2[:,:,0:3], pic[:,:,0:3])
            errors.append(error)
    i = np.argmin(errors)
    tgt = files[i]
    labelfile = f"{mydir}/{str(tgt.stem)}.txt"
    if path(labelfile).exists() == False:
        print(f"Warning: no labels found for {tgt}")
        return ""
    with open(labelfile,'r') as f:
        label = f.read()
    return label

def convert1024(mydir, srcdir = "512", tgtdir = "1024", do_resize = True, offset = 0):
    path(tgtdir).mkdir_p()
    for i,f in enumerate(path(mydir).files()):
        j = i + offset
        img = cv2.imread(f)
        if img is None:
            print("Error in file {}".format(str(f)))
            continue
        img512 = resize(img, 512)
        label = find_labels(img512, srcdir)
        if do_resize:
            img1024 = resize(img, 1024)
        else:
            img1024 = img
        base = f"{tgtdir}/{j}"
        cv2.imwrite(f"{base}.jpg", img1024)
        with open(f"{base}.txt",'w') as ff:
            ff.write(label)

def autocopytraining(mydir = "./", destdir = "512", traindir = "512"):
    for d in path(mydir).dirs():
        traindir2 = path(f"{d}/{traindir}")
        if not traindir2.exists():
            continue
        destdir2 = path(f"{destdir}/{str(d.name)}")
        destdir2.mkdir_p()
        for f in traindir2.files():
            f.copy(destdir2)

def zip_folder(tgtdir = "./"):
    tmpdir = tempfile.TemporaryDirectory()
    dest = f"{tmpdir.name}/pics"
    path(dest).mkdir_p()
    for f in path(tgtdir).files():
        img = cv2.imread(f)
        if img is not None:
            destfile = f"{dest}/{str(f.name)}"
            cv2.imwrite(destfile, img)
    zip_folder_pyzipper(dest, "./images.zip")
    tmpdir.cleanup()

def zip_folder_pyzipper(folder_path, output_path):
    """Zip the contents of an entire folder (with that folder included
    in the archive). Empty subfolders will be included in the archive
    as well.
    """
    # Retrieve the paths of the folder contents.
    contents = os.walk(folder_path)
    try:
        zip_file = pyzipper.AESZipFile(output_path, 'w', compression=pyzipper.ZIP_DEFLATED)
        for root, folders, files in contents:
            for folder_name in folders:
                absolute_path = os.path.join(root, folder_name)
                relative_path = os.path.relpath(absolute_path, start=folder_path)
                zip_file.write(absolute_path, relative_path)
            for file_name in files:
                absolute_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(absolute_path, start=folder_path)
                zip_file.write(absolute_path, relative_path)

        print(f"'{output_path}' created successfully.")

    except IOError as message:
        print(message)
        sys.exit(1)
    except OSError as message:
        print(message)
        sys.exit(1)
    except zipfile.BadZipfile as message:
        print(message)
        sys.exit(1)
    finally:
        zip_file.close()

def blur_image(image, ksize=(199,199)):
    """
    Applies Gaussian blur to the image at the specified path.

    Parameters:
    - image_path (str): Path to the input image.
    - ksize (tuple): Kernel size for the Gaussian blur. Must be odd integers.

    Returns:
    - blurred_image (numpy.ndarray): Blurred image.
    """
    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX=0)

    return blurred_image

def add_grain(
    image,
    strength=0.05,
    grain_type="gaussian",
    monochrome=False,
    selective=False
):
    """
    Adds grain to an image to reduce cartoon-like smoothness.

    Parameters:
        image (np.ndarray): Input BGR image in uint8 format (0-255).
        strength (float): Grain strength, typical range [0.01, 0.1].
        grain_type (str): 'gaussian' or 'uniform' noise.
        monochrome (bool): If True, applies the same noise to all color channels.
        selective (bool): If True, applies more grain to dark areas.

    Returns:
        np.ndarray: Grainy image in uint8 format.
    """

    # Convert to float32 in [0, 1]
    if isinstance(image, str):
        image = cv2.imread(image)
    img = image.astype(np.float32) / 255.0
    h, w, c = img.shape

    # Generate noise
    if monochrome:
        if grain_type == "gaussian":
            noise = np.random.normal(0.0, strength, size=(h, w))
        elif grain_type == "uniform":
            noise = np.random.uniform(-strength, strength, size=(h, w))
        else:
            raise ValueError("grain_type must be 'gaussian' or 'uniform'")
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    else:
        if grain_type == "gaussian":
            noise = np.random.normal(0.0, strength, size=(h, w, c))
        elif grain_type == "uniform":
            noise = np.random.uniform(-strength, strength, size=(h, w, c))
        else:
            raise ValueError("grain_type must be 'gaussian' or 'uniform'")

    # Optionally weight noise by luminance (for shadows)
    if selective:
        luminance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        weight = 1.0 - luminance  # More grain in dark areas
        weight = cv2.GaussianBlur(weight, (5, 5), 0)
        noise *= weight[:, :, np.newaxis]

    # Apply and clip
    grainy = np.clip(img + noise, 0.0, 1.0)

    return (grainy * 255).astype(np.uint8)

def create_meme_text_image_opencv(text, max_width_pixels, output_path=None):
    # Settings
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (255, 255, 255, 255)  # White text
    stroke_color = (0, 0, 0, 255)      # Black outline
    stroke_thickness = 2
    padding = 10  # Optional
    initial_font_scale = 1.0

    # Function to measure text size
    def get_text_size(scale):
        thickness = max(int(scale * 2), 1)  # Thickness proportional to font scale
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        return w, h, baseline

    # Binary search for max font scale
    low, high = 0.1, 10.0
    best_scale = None
    best_size = (0, 0, 0)
    
    for _ in range(20):  # 20 iterations is more than enough
        mid = (low + high) / 2
        size = get_text_size(mid)
        if size[0] <= max_width_pixels:
            best_scale = mid
            best_size = size
            low = mid
        else:
            high = mid

    if best_scale is None:
        raise ValueError("Text cannot fit within the given pixel width.")

    text_width, text_height, baseline = best_size
    img_width = text_width + 2 * padding
    img_height = text_height + baseline + 2 * padding

    # Create transparent image
    img = np.zeros((img_height, img_width, 4), dtype=np.uint8)

    # Center text
    text_x = padding
    text_y = padding + text_height

    # Draw outline (stroke)
    for dx in [-stroke_thickness, 0, stroke_thickness]:
        for dy in [-stroke_thickness, 0, stroke_thickness]:
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text, (text_x + dx, text_y + dy), font, best_scale, stroke_color, thickness=stroke_thickness, lineType=cv2.LINE_AA)

    # Draw main text
    cv2.putText(img, text, (text_x, text_y), font, best_scale, font_color, thickness=stroke_thickness, lineType=cv2.LINE_AA)

    # Save or return
    if output_path:
        cv2.imwrite(output_path, img)
    return img

def has_exif(img):
    return "exif" in img.info

def strip_jpeg_metadata(path_in, path_out):
    try:
        img = Image.open(path_in)
        if not has_exif(img):
            print(f"[SKIP] No EXIF in: {path_in}")
            return

        img_copy = Image.new(img.mode, img.size)
        img_copy.putdata(list(img.getdata()))
        img_copy.save(path_out)  # No EXIF param => EXIF segment removed
        print(f"[STRIPPED] EXIF removed: {path_in}")
    except Exception as e:
        print(f"[ERROR] {path_in}: {e}")

def strip_png_metadata(path_in, path_out):
    try:
        img = Image.open(path_in)
        if not isinstance(img.info, dict) or not img.info:
            print(f"[SKIP] No metadata in: {path_in}")
            return
        clean_img = Image.new(img.mode, img.size)
        clean_img.putdata(list(img.getdata()))
        clean_img.save(path_out)
        print(f"[STRIPPED] PNG metadata removed: {path_in}")
    except Exception as e:
        print(f"[ERROR] {path_in}: {e}")

def remove_metadata(p):
    ext = os.path.splitext(p)[1].lower()
    path_out = p  # Overwrite in place
    if ext in [".jpg", ".jpeg"]:
        strip_jpeg_metadata(p, path_out)
    elif ext == ".png":
        strip_png_metadata(p, path_out)
    else:
        print(f"[SKIP] Unsupported file type: {p}")

def remove_metadata_dir(p):
    [remove_metadata(a) for a in path(p).walkfiles()]

def split_916(imgfile, destdir = "./"):
    img = cv2.imread(imgfile)
    h,w = img.shape[0:2]
    a = int(round(w*16/9))
    b = int(h-a)
    img1 = img[0:a,:,:]
    img2 = img[b:h,:,:]
    fstem = str(path(imgfile).stem)
    base = f"{destdir}/{fstem}_"
    f1 = f"{base}_a.jpg"
    f2 = f"{base}_b.jpg"
    cv2.imwrite(f1, img1)
    cv2.imwrite(f2, img2)
    return f1, f2