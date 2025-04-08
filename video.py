import os
from moviepy.editor import ImageClip, ImageSequenceClip, concatenate_videoclips, CompositeVideoClip, VideoFileClip, AudioFileClip
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

def create_slideshow_with_single_audio(directory, image_duration, audio_file, zoom_factor=0.04, output=None):
    # Step 1: Create the slideshow part
    image_files = [f for f in path(directory).files() if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if not image_files:
        raise ValueError("No image files found in the specified directory.")
    if not output:
        output = str(path(directory).name).title() + ".mp4"
    if path(output).exists():
        print("Error: output file already exists!")
        return
    slides = [mp.ImageClip(str(url)).set_fps(25).set_duration(image_duration) for url in image_files]
    slides = [zoom_in_effect(slide, zoom_factor) for slide in slides]  # Apply zoom effect
    video_clip = concatenate_videoclips(slides)

    # Step 2: Validate the audio file
    if not os.path.exists(audio_file):
        raise ValueError(f"Audio file '{audio_file}' not found.")
    if not audio_file.endswith(('.mp3', '.wav')):
        raise ValueError("Audio file must be in .mp3 or .wav format.")

    # Step 3: Integrate the audio with the video
    try:
        final_audio = AudioFileClip(audio_file)  # Load the audio file
        video_duration = video_clip.duration  # Get video duration in seconds
        audio_duration = final_audio.duration  # Get audio duration in seconds

        # Truncate or pad the video duration to match the audio duration if necessary
        final_clip = video_clip.set_audio(final_audio.set_duration(min(video_duration, audio_duration)))

        # Step 4: Write the final video with audio
        final_clip.write_videofile(output, codec="libx264", audio_codec="aac")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Slideshow with audio saved as {output}")

def create_slideshow(directory, image_duration, zoom_factor=0.04, randomize=False, output=None, intro=None):
    image_files = get_image_files(directory)
    if not output:
        output = str(path(directory).name).title() + ".mp4"
    if path(output).exists():
        print("Error: output file already exists!")
        return

    if not image_files:
        print("No image files found in the specified directory.")
        return

    if randomize:
        random.shuffle(image_files)

    slides = []
    total_images = len(image_files)

    intro_duration, intro_count = (None, 0)
    if intro:
        intro_duration, intro_count = intro
        if intro_count > total_images:
            intro_count = total_images

    for n, url in enumerate(image_files):
        duration = intro_duration if intro and n < intro_count else image_duration
        clip = mp.ImageClip(url).set_fps(25).set_duration(duration)
        slides.append(zoom_in_effect(clip, zoom_factor))

    video = mp.concatenate_videoclips(slides)
    video.write_videofile(output)

def create_slideshow_with_intro_vid(directory, image_duration, intro_video_path, zoom_factor=0.04, randomize=False, output=None):
    """
    Creates a slideshow from images and prepends an intro video.

    Parameters:
        directory (str): Path to the directory containing images.
        image_duration (float): Duration each image is shown.
        intro_video_path (str): Path to the intro video file.
        zoom_factor (float): Amount of zoom-in effect per second.
        randomize (bool): Whether to shuffle the images.
        output (str): Output video filename. If None, uses folder name.
    """
    image_files = get_image_files(directory)

    if not output:
        output = str(path(directory).name).title() + ".mp4"
    if path(output).exists():
        print("Error: output file already exists!")
        return
    if not image_files:
        print("No image files found in the specified directory.")
        return
    if not os.path.isfile(intro_video_path):
        print(f"Intro video not found: {intro_video_path}")
        return

    if randomize:
        random.shuffle(image_files)

    # Create zoom-in clips from images
    slides = []
    for url in image_files:
        clip = mp.ImageClip(url).set_fps(25).set_duration(image_duration)
        slides.append(zoom_in_effect(clip, zoom_factor))

    slideshow_clip = mp.concatenate_videoclips(slides)

    # Load the intro video
    intro_clip = VideoFileClip(intro_video_path)

    # Match dimensions if needed (optional: skip this if intro and slideshow always match)
    if intro_clip.size != slideshow_clip.size:
        slideshow_clip = slideshow_clip.resize(newsize=intro_clip.size)

    # Concatenate intro + slideshow
    final_clip = concatenate_videoclips([intro_clip, slideshow_clip], method="compose")

    # Write the output video
    final_clip.write_videofile(output, codec="libx264", audio_codec="aac")


def create_slideshow_old(directory, image_duration, zoom_factor=0.04, randomize = False, output = None):
    image_files = get_image_files(directory)
    if not output:
        output = str(path(directory).name).title() + ".mp4"
    if path(output).exists():
        print("Error: output file already exists!")
        return

    if not image_files:
        print("No image files found in the specified directory.")
        return
    if randomize:
        random.shuffle(image_files)
    slides = []
    for n, url in enumerate(image_files):
        slides.append(
            mp.ImageClip(url).set_fps(25).set_duration(image_duration)
        )

        slides[n] = zoom_in_effect(slides[n], zoom_factor)


    video = mp.concatenate_videoclips(slides)    
    video.write_videofile(output)

def create_slideshow_with_audio(directory, image_duration, mp3_directory, zoom_factor=0.04, output=None, randomize=False):
    # Step 1: Get image files and create slideshow
    image_files = [f for f in path(directory).files() if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    if not image_files:
        raise ValueError("No image files found in the specified directory.")
    if randomize:
        random.shuffle(image_files)
    if not output:
        output = str(path(directory).name).title() + ".mp4"
    if path(output).exists():
        print("Error: output file already exists!")
        return

    slides = [mp.ImageClip(str(url)).set_fps(25).set_duration(image_duration) for url in image_files]
    slides = [zoom_in_effect(slide, zoom_factor) for slide in slides]  # Apply zoom effect
    video_clip = concatenate_videoclips(slides)
    video_duration = video_clip.duration  # Get video duration in seconds

    # Step 2: Prepare the audio track
    mp3_files = [os.path.join(mp3_directory, file) for file in os.listdir(mp3_directory)
                 if file.endswith(('.mp3', '.wav', '.mp4'))]

    if not mp3_files:
        raise ValueError("No MP3 or audio files found in the specified directory.")

    # Load and trim audio
    selected_mp3 = random.choice(mp3_files)  # Randomly pick an MP3 file
    audio_clip = AudioSegment.from_file(selected_mp3)

    # Trim the audio to match video length exactly
    trimmed_audio = audio_clip[:int(video_duration * 1000)]

    # Step 3: Save trimmed audio temporarily
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_audio_path = os.path.join(tempdir, "temp_audio.mp3")
            trimmed_audio.export(temp_audio_path, format="mp3", bitrate="192k")

            # Load into moviepy
            final_audio = AudioFileClip(temp_audio_path)

            # Ensure audio matches video duration exactly
            final_clip = video_clip.set_audio(final_audio.subclip(0, video_duration))

            # Step 4: Write final video with audio
            final_clip.write_videofile(output, codec="libx264", audio_codec="aac")
    except Exception as e:
        print(f"Error: {e}")

    print(f"Slideshow with audio saved as {output}")


def bulk_slideshow(directory = "./", minduration = 61, minframe = 3, zoom_factor=0.04, existing = []):
    existing_set = set(existing)
    folders = path(directory).dirs()
    for d in folders:
        if d in existing_set:
            continue
        imgfiles = get_image_files(d)
        if len(imgfiles)*minframe < minduration:
            dur = minduration / len(imgfiles)
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

def append_flash_image(video_path, image_path, flash_duration, output_path="output_with_flash.mp4"):
    """
    Appends a brief flash of an image at the end of a video using MoviePy.
    
    Parameters:
        video_path (str): Path to the input video file.
        image_path (str): Path to the image to flash at the end.
        flash_duration (float): Duration (in seconds) for the image flash.
        output_path (str): Path to save the output video file.
        
    Returns:
        str: The output file path.
    """
    # Load the main video
    video_clip = VideoFileClip(video_path)
    
    # Create an image clip with the specified duration
    image_clip = ImageClip(image_path).set_duration(flash_duration)
    
    # Resize the image clip to match the video’s dimensions and set the frame rate
    image_clip = image_clip.resize(newsize=video_clip.size)
    image_clip = image_clip.set_fps(video_clip.fps)
    
    # Concatenate the video and image clips using the "compose" method to handle mismatches in frames or audio
    final_clip = concatenate_videoclips([video_clip, image_clip], method="compose")
    
    # Write the final video, explicitly matching the original video's fps and codecs
    final_clip.write_videofile(
        output_path,
        fps=video_clip.fps,
        codec="libx264",
        audio_codec="aac"
    )
    
    return output_path

def create_image_video(image_path, duration, output_path="output_image_video.mp4", fps=25):
    """
    Creates a video from a single image displayed for a specified duration.

    Parameters:
        image_path (str): Path to the input image.
        duration (float): Duration in seconds for the image to be shown.
        output_path (str): Path to save the output video.
        fps (int): Frames per second of the output video.

    Returns:
        str: The output file path.
    """
    try:
        clip = ImageClip(image_path).set_duration(duration).set_fps(fps)
        clip.write_videofile(output_path, codec="libx264", audio=False, fps=fps)
        return output_path
    except Exception as e:
        print(f"Error: {e}")
        return None