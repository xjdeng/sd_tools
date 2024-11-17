import os
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, CompositeVideoClip, VideoFileClip, AudioFileClip
from moviepy.editor import vfx
from pydub import AudioSegment
import moviepy.editor as mp
import math
from PIL import Image
import numpy
from path import Path as path
import math
import random
import tempfile

def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]

        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        img = img.resize(new_size, Image.LANCZOS)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)

        result = numpy.array(img)
        img.close()

        return result

    return clip.fl(effect)

def get_image_files(directory):
    return [f for f in path(directory).files() if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

def create_slideshow(directory, image_duration, zoom_factor=0.04, output = "zoomin.mp4"):
    image_files = get_image_files(directory)

    if not image_files:
        print("No image files found in the specified directory.")
        return
    slides = []
    for n, url in enumerate(image_files):
        slides.append(
            mp.ImageClip(url).set_fps(25).set_duration(image_duration)
        )

        slides[n] = zoom_in_effect(slides[n], zoom_factor)


    video = mp.concatenate_videoclips(slides)    
    video.write_videofile(output)

def create_slideshow_with_audio(directory, image_duration, mp3_directory, zoom_factor=0.04, output= None):
    # Step 1: Create the slideshow part
    image_files = [f for f in path(directory).files() if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if not image_files:
        raise ValueError("No image files found in the specified directory.")
    if not output:
        output = str(path(directory).name) + ".mp4"
    if path(output).exists():
        print("Error: output file already exists!")
        return
    slides = [mp.ImageClip(str(url)).set_fps(25).set_duration(image_duration) for url in image_files]
    slides = [zoom_in_effect(slide, zoom_factor) for slide in slides]  # Apply zoom effect
    video_clip = concatenate_videoclips(slides)

    # Step 2: Generate the audio part
    video_duration = video_clip.duration  # Get video duration in seconds
    mp3_files = [os.path.join(mp3_directory, file) for file in os.listdir(mp3_directory) if file.endswith('.mp3')]

    if not mp3_files:
        raise ValueError("No MP3 files found in the specified directory.")

    # Prepare the audio track
    audio_segments = []
    current_duration = 0
    while current_duration < video_duration:
        if not mp3_files:  # Reset mp3 files if exhausted
            mp3_files = [os.path.join(mp3_directory, file) for file in os.listdir(mp3_directory) if file.endswith('.mp3')]
        random_mp3 = random.choice(mp3_files)
        mp3_files.remove(random_mp3)  # Avoid repetition
        audio_clip = AudioSegment.from_file(random_mp3)

        if current_duration + audio_clip.duration_seconds <= video_duration:
            audio_segments.append(audio_clip)
            current_duration += audio_clip.duration_seconds
        else:
            remaining_duration = video_duration - current_duration
            audio_segments.append(audio_clip[:int(remaining_duration * 1000)])
            break

    combined_audio = sum(audio_segments)

    # Step 3: Save combined audio temporarily and integrate it with video
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_audio_path = os.path.join(tempdir, "temp_audio.mp3")
            combined_audio.export(temp_audio_path, format="mp3")
            final_audio = AudioFileClip(temp_audio_path)
            final_clip = video_clip.set_audio(final_audio)

            # Step 4: Write final video with audio

            final_clip.write_videofile(output, codec="libx264", audio_codec="aac")
    except NotADirectoryError:
        pass
    print(f"Slideshow with audio saved as {output}")

def bulk_slideshow(directory = "./", minduration = 61, minframe = 3, zoom_factor=0.04, existing = []):
    existing_set = set(existing)
    folders = path(directory).dirs()
    for d in folders:
        if d in existing_set:
            continue
        imgfiles = get_image_files(d)
        if len(imgfiles)*minframe < minduration:
            dur = math.ceil(minduration / len(imgfiles))
        else:
            dur = minframe
        outfile = f"{str(d.name)}.mp4"
        if not path(outfile).exists():
            create_slideshow(d, dur, zoom_factor, output = outfile)

def add_random_audio_to_video(source_video_path, target_video_path, mp3_directory):
    # Load the video to get its duration
    video_clip = VideoFileClip(source_video_path)
    video_duration = video_clip.duration  # Duration in seconds

    # Get a list of all mp3 files in the directory
    mp3_files_original = [os.path.join(mp3_directory, file) for file in os.listdir(mp3_directory) if file.endswith('.mp3')]
    
    # Ensure there are mp3 files available
    if not mp3_files_original:
        raise ValueError("No MP3 files found in the specified directory.")
    
    # Create a working copy of the mp3 file list
    mp3_files = mp3_files_original.copy()
    audio_segments = []
    current_duration = 0

    while current_duration < video_duration:
        if not mp3_files:  # If the list is exhausted, replenish it
            mp3_files = mp3_files_original.copy()

        random_mp3 = random.choice(mp3_files)
        mp3_files.remove(random_mp3)  # Remove the chosen file to avoid immediate repetition
        audio_clip = AudioSegment.from_file(random_mp3)

        if current_duration + audio_clip.duration_seconds <= video_duration:
            audio_segments.append(audio_clip)
            current_duration += audio_clip.duration_seconds
        else:
            # If adding the full clip exceeds the duration, add only a segment of it
            remaining_duration = video_duration - current_duration
            trimmed_clip = audio_clip[:remaining_duration * 1000]  # Convert seconds to milliseconds
            audio_segments.append(trimmed_clip)
            break

    # Combine all audio segments into a single audio track
    combined_audio = sum(audio_segments)

    # Export the combined audio as a temporary file
    tempdir = tempfile.TemporaryDirectory()
    temp_audio_path = f"{tempdir.name}/temp_audio.mp3"
    combined_audio.export(temp_audio_path, format="mp3")

    # Load the combined audio using moviepy
    final_audio = AudioFileClip(temp_audio_path)

    # Set the audio of the video clip to the combined audio
    final_clip = video_clip.set_audio(final_audio)

    # Export the final video
    final_clip.write_videofile(target_video_path, codec="libx264", audio_codec="aac")

    # Cleanup temporary file
    try:
        tempdir.cleanup()
    except NotADirectoryError:
        pass

    print(f"Video saved as {target_video_path}")

def bulk_add_random_audio(source_dir, dest_dir, mp3_directory):
    files = path(source_dir).files()
    path(dest_dir).mkdir_p()
    for f in files:
        dest = f"{dest_dir}/{str(f.name)}"
        if not path(dest).exists():
            try:
                add_random_audio_to_video(f, dest, mp3_directory)
            except Exception as e:
                print(f"Error {e} in file {str(f)}")
