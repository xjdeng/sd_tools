import os
from PIL import Image
from moviepy.editor import VideoFileClip

def remove_image_metadata(input_path, output_path):
    with Image.open(input_path) as img:
        data = list(img.getdata())
        img_no_metadata = Image.new(img.mode, img.size)
        img_no_metadata.putdata(data)
        img_no_metadata.save(output_path)

def remove_video_metadata(input_path, output_path):
    try:
        clip = VideoFileClip(input_path)
        # Write video without audio and metadata
        clip.write_videofile(output_path, audio=True, codec='libx264', remove_temp=True, preset='medium', threads=4)
        clip.close()
    except Exception as e:
        print(f"Error processing video {input_path}: {e}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        ext = os.path.splitext(filename)[-1].lower()
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                remove_image_metadata(input_path, output_path)
                print(f"Image processed: {filename}")
            elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
                remove_video_metadata(input_path, output_path)
                print(f"Video processed: {filename}")
            else:
                print(f"Skipped (unsupported format): {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
