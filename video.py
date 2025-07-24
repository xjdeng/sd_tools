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
import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
import subprocess

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
    from moviepy.editor import VideoFileClip, AudioFileClip
    from pydub import AudioSegment
    import tempfile
    import os
    import random

    # Load the video to get its duration
    video_clip = VideoFileClip(source_video_path)
    video_duration = video_clip.duration  # Duration in seconds

    # Get all MP3 files in the directory
    mp3_files_original = [os.path.join(mp3_directory, file)
                          for file in os.listdir(mp3_directory) if file.endswith('.mp3')]

    if not mp3_files_original:
        raise ValueError("No MP3 files found in the specified directory.")

    mp3_files = mp3_files_original.copy()
    audio_segments = []
    current_duration = 0

    while current_duration < video_duration:
        if not mp3_files:
            mp3_files = mp3_files_original.copy()

        random_mp3 = random.choice(mp3_files)
        mp3_files.remove(random_mp3)
        audio_clip = AudioSegment.from_file(random_mp3)

        if current_duration + audio_clip.duration_seconds <= video_duration:
            audio_segments.append(audio_clip)
            current_duration += audio_clip.duration_seconds
        else:
            remaining_duration = video_duration - current_duration
            trimmed_clip = audio_clip[:int(remaining_duration * 1000)]
            audio_segments.append(trimmed_clip)
            break

    # Combine all segments
    combined_audio = sum(audio_segments)

    # Save to temporary MP3
    with tempfile.TemporaryDirectory() as tempdir:
        temp_audio_path = os.path.join(tempdir, "temp_audio.mp3")
        combined_audio.export(temp_audio_path, format="mp3", bitrate="192k")

        # Load and force duration match
        final_audio = AudioFileClip(temp_audio_path).subclip(0, video_duration)

        # Apply audio to video and export
        final_clip = video_clip.set_audio(final_audio)
        final_clip.write_videofile(target_video_path, codec="libx264", audio_codec="aac")

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
    


def fix_input_video(input_path):
    """
    Re-encode a potentially broken video file to a standard format.
    This function uses ffmpeg to convert the input to H.264 (baseline profile),
    with yuv420p pixel format, AAC audio at 192k, and faststart metadata for mobile compatibility.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        fixed_path = tmp.name
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        fixed_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return fixed_path

def join_videos_resize_second(video1_path, video2_path, output_path="joined_video.mp4"):
    """
    Fixes and joins two potentially broken video clips into one file.
    
    Each input clip is first fixed via ffmpeg to create a standard, Instagram-friendly file.
    Then, using MoviePy, each fixed clip is re-encoded to a temporary file with uniform parameters
    (using the dimensions and FPS of the first clip). Finally, ffmpeg’s concat demuxer joins the clips.
    
    Parameters:
        video1_path (str): Path to the first (possibly broken) video.
        video2_path (str): Path to the second (possibly broken) video.
        output_path (str): Path to write the final joined video.
        
    Returns:
        str: The output file path on success, or None if an error occurs.
    """
    fixed1_path = None
    fixed2_path = None
    temp1_path = None
    temp2_path = None
    list_filename = None

    try:
        # Step 1: Fix the input videos.
        fixed1_path = fix_input_video(video1_path)
        fixed2_path = fix_input_video(video2_path)
        
        # Step 2: Load fixed videos with MoviePy.
        clip1 = VideoFileClip(fixed1_path)
        clip2 = VideoFileClip(fixed2_path)
        
        # Use the dimensions and FPS of clip1 as the reference.
        target_size = clip1.size  # [width, height]
        target_fps = clip1.fps
        
        # Resize clip2 if its dimensions differ; match its FPS if needed.
        if clip2.size != target_size:
            clip2 = clip2.resize(newsize=target_size)
        if clip2.fps != target_fps:
            clip2 = clip2.set_fps(target_fps)
        
        # Helper: Ensure the clip has an audio track spanning its full duration.
        def ensure_audio(clip, audio_fps=44100):
            if clip.audio is None:
                duration = clip.duration
                num_samples = int(duration * audio_fps)
                silent_audio_array = np.zeros((num_samples, 2), dtype=np.float32)  # stereo silence
                silent_audio = AudioArrayClip(silent_audio_array, fps=audio_fps)
                return clip.set_audio(silent_audio)
            elif clip.audio.duration < clip.duration:
                return clip.set_audio(clip.audio.set_duration(clip.duration))
            return clip
        
        clip1 = ensure_audio(clip1)
        clip2 = ensure_audio(clip2)
        
        # Step 3: Re-encode each clip to a temporary file with uniform parameters.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp1_path = tmp.name
        clip1.write_videofile(
            temp1_path,
            codec="libx264",
            audio_codec="aac",
            fps=target_fps,
            preset="medium",
            audio_bitrate="192k",
            verbose=False,
            logger=None
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp2_path = tmp.name
        clip2.write_videofile(
            temp2_path,
            codec="libx264",
            audio_codec="aac",
            fps=target_fps,
            preset="medium",
            audio_bitrate="192k",
            verbose=False,
            logger=None
        )
        
        # Step 4: Use ffmpeg’s concat demuxer to join the two temporary files.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file_list:
            list_filename = file_list.name
            file_list.write(f"file '{temp1_path}'\n")
            file_list.write(f"file '{temp2_path}'\n")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_filename,
            "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        
        return output_path
        
    except Exception as e:
        print(f"Error joining videos: {e}")
        return None
    finally:
        # Cleanup all temporary files.
        for f in [fixed1_path, fixed2_path, temp1_path, temp2_path, list_filename]:
            if f and os.path.exists(f):
                os.remove(f)

def insert_still_frame(video_path, image_path, insert_time=None, duration=0.25, output_path="output_with_insert.mp4"):
    """
    Inserts a still image into the middle of a video at a given timestamp.

    Parameters:
        video_path (str):       Path to the input video file.
        image_path (str):       Path to the image to insert.
        insert_time (float|None): Time (in seconds) to insert the image. If None, picks a random spot.
        duration (float):       How many seconds the still frame should last (default 0.25).
        output_path (str):      Path to save the output video (default "output_with_insert.mp4").

    Returns:
        str: The output file path.
    """
    # Load the source video and drop its audio
    clip = VideoFileClip(video_path)
    clip = clip.set_audio(None)
    video_duration = clip.duration
    fps = clip.fps
    size = clip.size

    # Decide where to insert
    if insert_time is None:
        insert_time = random.uniform(0, video_duration)
    else:
        # clamp to [0, video_duration]
        insert_time = max(0, min(insert_time, video_duration))

    # Prepare the image clip
    img_clip = ImageClip(image_path).set_duration(duration).set_fps(fps)
    img_clip = img_clip.resize(newsize=size)

    # Split the video
    before = clip.subclip(0, insert_time)
    after  = clip.subclip(insert_time, video_duration)

    # Concatenate: before + image + after
    final = concatenate_videoclips([before, img_clip, after], method="compose")

    # Write out with no audio
    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio=False
    )

    return output_path

def _change_hue(clip, hue_shift=None):
    """
    Change hue of the video clip by hue_shift degrees (0-360).
    If hue_shift is None, choose a random shift each run.
    """
    if hue_shift is None:
        hue_shift = random.uniform(-20, 20)  # small random hue change in degrees

    def shift_hue(frame):
        # Convert frame (H x W x 3) to float32
        frame = frame.astype(np.float32) / 255.0
        # Convert to HSV
        hsv = np.zeros_like(frame)
        # RGB -> HSV
        r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        maxc = np.max(frame, axis=2)
        minc = np.min(frame, axis=2)
        v = maxc
        delta = maxc - minc
        s = delta / (maxc + 1e-8)
        
        # Hue calculation
        hue = np.zeros_like(maxc)
        mask = delta > 1e-8
        idx = (maxc == r) & mask
        hue[idx] = (g[idx] - b[idx]) / delta[idx]
        idx = (maxc == g) & mask
        hue[idx] = 2.0 + (b[idx] - r[idx]) / delta[idx]
        idx = (maxc == b) & mask
        hue[idx] = 4.0 + (r[idx] - g[idx]) / delta[idx]
        hue = (hue / 6.0) % 1.0
        
        # Shift hue
        hue = (hue + hue_shift / 360.0) % 1.0
        
        # HSV -> RGB
        i = np.floor(hue * 6).astype(int)
        f = hue * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r2, g2, b2 = np.zeros_like(hue), np.zeros_like(hue), np.zeros_like(hue)
        
        i_mod = i % 6
        r2[i_mod == 0], g2[i_mod == 0], b2[i_mod == 0] = v[i_mod == 0], t[i_mod == 0], p[i_mod == 0]
        r2[i_mod == 1], g2[i_mod == 1], b2[i_mod == 1] = q[i_mod == 1], v[i_mod == 1], p[i_mod == 1]
        r2[i_mod == 2], g2[i_mod == 2], b2[i_mod == 2] = p[i_mod == 2], v[i_mod == 2], t[i_mod == 2]
        r2[i_mod == 3], g2[i_mod == 3], b2[i_mod == 3] = p[i_mod == 3], q[i_mod == 3], v[i_mod == 3]
        r2[i_mod == 4], g2[i_mod == 4], b2[i_mod == 4] = t[i_mod == 4], p[i_mod == 4], v[i_mod == 4]
        r2[i_mod == 5], g2[i_mod == 5], b2[i_mod == 5] = v[i_mod == 5], p[i_mod == 5], q[i_mod == 5]
        
        out = np.stack([r2, g2, b2], axis=2)
        out = (out * 255).astype(np.uint8)
        return out

    return clip.fl_image(shift_hue)

def change_hue(clip, disable_audio=True):
    hue_shift = random.uniform(-20,20)
    identity = str(abs(hue_shift)).replace(".","")
    clip_dir = path(clip).abspath().dirname()
    clip_name = path(clip).stem
    clip_ext = path(clip).ext
    outfile = f"{clip_dir}/{clip_name}_{identity}{clip_ext}"
    myclip = VideoFileClip(clip)
    clip_hue = _change_hue(myclip, hue_shift)
    clip_hue.write_videofile(outfile, codec="libx264", audio=not disable_audio)