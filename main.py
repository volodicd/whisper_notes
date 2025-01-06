import os
import subprocess
import ffmpeg
from gpt4all import GPT4All
from PIL import Image
from tqdm import tqdm


#############################
# 1. Download YouTube Video #
#############################

def download_youtube_video (youtube_url, output_filename="lecture.mp4"):
    """
    Downloads a YouTube video using yt-dlp.
    """
    print (f"Starting download of YouTube video: {youtube_url}")
    try:
        # Remove file extension from output_filename
        output_template = os.path.splitext (output_filename)[0]

        subprocess.run (
            ["yt-dlp",
             youtube_url,
             "-o", output_template + ".%(ext)s",
             "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
             "--merge-output-format", "mp4"
             ],
            check=True
        )

        final_output = output_template + ".mp4"
        if os.path.exists (final_output):
            print (f"Download completed. Video saved as: {final_output}")
            return final_output
        else:
            print ("Error: Output file not found after download")
            return None

    except subprocess.CalledProcessError as e:
        print ("Error downloading video:", e)
        return None

################################
# 2. Extract Audio from Video  #
################################

def extract_audio (video_path, audio_path):
    """
    Extract the audio from the video file and save as a WAV using ffmpeg.
    """
    print (f"Extracting audio from {video_path} ...")
    try:
        # First check if the input file exists
        if not os.path.exists (video_path):
            print (f"Error: Input file {video_path} does not exist")
            return None

        # Create the ffmpeg stream
        stream = ffmpeg.input (video_path)
        stream = ffmpeg.output (stream, audio_path, acodec='pcm_s16le', ac=2)

        # Run ffmpeg
        try:
            ffmpeg.run (stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            print (f"Audio saved to {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            if e.stderr:
                print ("FFmpeg error:", e.stderr.decode ('utf-8'))
            else:
                print ("An error occurred while running FFmpeg")
            return None

    except Exception as e:
        print ("Error during audio extraction:", str (e))
        return None


#########################################
# 3. Keyframe Extraction               #
#########################################

def extract_keyframes (video_path, output_dir, interval=10, use_scene_cut=False):
    """
    Extract frames from the video at a regular interval using ffmpeg.
    """
    print ("Extracting keyframes...")
    os.makedirs (output_dir, exist_ok=True)

    try:
        # Get video duration using ffprobe
        probe = ffmpeg.probe (video_path)
        duration = float (probe['streams'][0]['duration'])

        for t in tqdm (range (0, int (duration), interval), desc="Keyframes"):
            output_file = os.path.join (output_dir, f"frame_{t}.jpg")
            stream = ffmpeg.input (video_path, ss=t)
            stream = ffmpeg.output (stream, output_file, vframes=1)
            ffmpeg.run (stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)

        print (f"Keyframes saved in {output_dir}")
        return output_dir
    except ffmpeg.Error as e:
        print ("Error during keyframe extraction:", e.stderr.decode ())
        return None


#######################################
# 4. Transcribe Audio with Whisper    #
#######################################

def transcribe_audio (audio_path, model_size="base"):
    """
    Transcribe the audio using the Whisper CLI.
    model_size can be: 'tiny', 'base', 'small', 'medium', 'large'
    (subject to the models you have installed).
    """
    print (f"Transcribing audio with Whisper ({model_size} model)...")
    try:
        # Create folder for transcriptions
        os.makedirs ("./transcriptions", exist_ok=True)

        result = subprocess.run (
            [
                "whisper",
                audio_path,
                "--model", model_size,
                "--output_dir", "./transcriptions",
                "--language", "en"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print ("Transcription completed.")
            transcript_path = os.path.splitext (os.path.basename (audio_path))[0] + ".txt"
            transcript_path = f"./transcriptions/{transcript_path}"
            with open (transcript_path, "r") as f:
                transcript = f.read ()
            return transcript
        else:
            print ("Error in transcription:", result.stderr)
            return None
    except Exception as e:
        print ("Error during transcription:", e)
        return None


#############################################################
# 5. Split Transcript into Chunks to Avoid LLM Context Limit #
#############################################################

def split_transcript_into_chunks (transcript, max_tokens_per_chunk=1500):
    """
    Splits the transcript into smaller chunks so we don't exceed
    the local LLM's context window.
    This is a basic approach by word count.
    You can refine by sentence boundaries or tokens if needed.
    """
    words = transcript.split ()
    chunks = []
    current_chunk = []

    current_tokens = 0
    for word in words:
        current_chunk.append (word)
        current_tokens += 1
        if current_tokens >= max_tokens_per_chunk:
            chunks.append (" ".join (current_chunk))
            current_chunk = []
            current_tokens = 0

    # Append the remainder
    if current_chunk:
        chunks.append (" ".join (current_chunk))

    return chunks


######################################################
# 6. Summarize Text Chunks with references to frames #
######################################################

def summarize_chunks_with_visuals(transcript_chunks, keyframes_dir, model_path=None):
    """
    Summarize each transcript chunk with GPT4All, referencing keyframes.
    Then combine the chunk-summaries into a final summarized note.
    """
    print("Summarizing transcript in chunks...")

    try:
        # Initialize GPT4All with your model
        model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        gpt = GPT4All(model_name)
        print(f"Initialized model: {model_name}")

        # Collect keyframes
        keyframes = []
        if os.path.isdir(keyframes_dir):
            keyframes = sorted(os.listdir(keyframes_dir))
        keyframe_references = "\n".join([f"- {frame}" for frame in keyframes])

        final_summaries = []
        for i, chunk in enumerate(tqdm(transcript_chunks, desc="Summarizing Chunks")):
            # Construct the prompt for each chunk
            prompt = (
                "You are a helpful AI. Summarize the following portion of a lecture transcript "
                "into clear, concise notes. Incorporate references to the extracted keyframes if relevant.\n\n"
                f"Keyframes:\n{keyframe_references}\n\n"
                f"Chunk {i + 1}:\n{chunk}\n\n"
                "Summary:\n"
            )

            # Generate a summary for the chunk with specific parameters
            summary = gpt.generate(
                prompt,
                max_tokens=512,
                temp=0.7,
                top_k=40,
                top_p=0.4,
                repeat_penalty=1.18,
                repeat_last_n=64,
                n_batch=8,
                n_predict=None
            )
            final_summaries.append(summary)

        # Once all chunks are summarized, combine them into one final set of notes
        combined_prompt = (
            "Combine the following chunk summaries into one comprehensive summary. "
            "Aim to keep it concise, well-organized, and clear:\n\n"
            + "\n\n".join([f"Chunk {i + 1} summary:\n{sm}" for i, sm in enumerate(final_summaries)])
            + "\n\nFinal combined summary:\n"
        )

        final_notes = gpt.generate(
            combined_prompt,
            max_tokens=1024,
            temp=0.7,
            top_k=40,
            top_p=0.4,
            repeat_penalty=1.18,
            repeat_last_n=64,
            n_batch=8,
            n_predict=None
        )
        return final_notes

    except Exception as e:
        print(f"Error in summarization: {e}")
        import traceback
        traceback.print_exc()
        return None

##########################################
# 7. Main Orchestration Function         #
##########################################

def main ():
    """
    Main flow:
      1. Ask user for YouTube URL
      2. Download the video
      3. Extract audio
      4. Extract frames
      5. Transcribe
      6. Chunk transcript
      7. Summarize chunked transcript with GPT4All
      8. Save final notes
    """

    # 1. Prompt for YouTube URL and download
    youtube_url = input ("Enter the YouTube URL for the lecture video: ")
    video_path = download_youtube_video (youtube_url, output_filename="lecture.mp4")
    if not video_path:
        print ("Download failed. Exiting.")
        return

    # 2. Extract audio
    audio_path = "temp_audio.wav"
    extract_audio (video_path, audio_path)

    # 3. Extract frames (every 10s, no scene detection for now)
    keyframes_dir = "./keyframes"
    extract_keyframes (video_path, keyframes_dir, interval=10, use_scene_cut=False)

    # 4. Transcribe (choose your Whisper model: 'base', 'medium', 'large', etc.)
    transcript = transcribe_audio (audio_path, model_size="medium")
    if not transcript:
        print ("Transcription failed. Exiting.")
        return

    # 5. Split the transcript into chunks
    transcript_chunks = split_transcript_into_chunks (transcript, max_tokens_per_chunk=1500)
    print (f"Transcript split into {len (transcript_chunks)} chunks.")

    # 6. Summarize chunked transcript with GPT4All
    # Make sure this path matches where your GPT4All model is located
    final_notes = summarize_chunks_with_visuals(transcript_chunks, keyframes_dir, None)  # model_path can be None
    if not final_notes:
        print("Summarization failed. Exiting.")
        return

    # 7. Save final notes
    notes_path = "lecture_notes_with_visuals.txt"
    with open (notes_path, "w") as f:
        f.write (final_notes)
    print (f"Notes saved to {notes_path}")

    # 8. Cleanup
    if os.path.exists (audio_path):
        os.remove (audio_path)
    # (Optionally remove the video if not needed)
    # os.remove(video_path)

    print ("Done.")


if __name__ == "__main__":
    main ()
