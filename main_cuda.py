import os
import subprocess
import ffmpeg
import torch
import torch.multiprocessing as mp
from gpt4all import GPT4All
from PIL import Image
from tqdm import tqdm
import whisper
import cv2
import numpy as np
from datetime import datetime
import joblib
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.cluster import KMeans
from transformers import pipeline
from concurrent.futures import TimeoutError
import logging

# Suppress CUDA-related warnings
import warnings

warnings.filterwarnings ("ignore", message="Failed to load libllamamodel")
torch.cuda.set_per_process_memory_fraction(0.9)

# Enable detailed logging for GPT4All
logging.getLogger ('gpt4all').setLevel (logging.DEBUG)

# ------------------------------------------------------------------------------------
# Global Settings
# ------------------------------------------------------------------------------------
# Force CPU mode
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['USE_CUDA'] = '0'
mp.set_start_method ('spawn', force=True)

CACHE_DIR = ".cache"
LOG_FILE = "video_processor.log"
MODEL_DIR = "models"
TEMP_AUDIO_FILE = "temp_audio.wav"
OUTPUT_PDF = "lecture_notes.pdf"
CHUNK_SIZE = 1000
NUM_WORKERS = 2
MIN_SCENE_LENGTH = 30
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Enable GPU usage
os.environ['USE_CUDA'] = '1'
DEFAULT_MODEL_NAME = "orca-mini-3b-gguf2-q4_0.gguf"

# ------------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------------
logging.basicConfig (
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler (LOG_FILE),
        logging.StreamHandler ()
    ]
)


# ------------------------------------------------------------------------------------
# CacheManager: Handles caching for repeated operations
# ------------------------------------------------------------------------------------
class CacheManager:
    def __init__ (self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs (cache_dir, exist_ok=True)

    def get_cache_key (self, data):
        """
        Creates an MD5 hash of stringified data to use as a cache key.
        """
        if isinstance (data, str):
            return hashlib.md5 (data.encode ()).hexdigest ()
        return hashlib.md5 (str (data).encode ()).hexdigest ()

    def get_cached_data (self, key, timeout_hours=24):
        """
        Retrieves data from cache if it exists and is not older than timeout_hours.
        """
        cache_file = os.path.join (self.cache_dir, f"{key}.joblib")
        if os.path.exists (cache_file):
            timestamp = os.path.getmtime (cache_file)
            if (datetime.now ().timestamp () - timestamp) < timeout_hours * 3600:
                return joblib.load (cache_file)
        return None

    def cache_data (self, key, data):
        """
        Caches data with joblib using the specified key.
        """
        cache_file = os.path.join (self.cache_dir, f"{key}.joblib")
        joblib.dump (data, cache_file)

    def cleanup_old_cache (self, max_age_hours=72):
        """
        Removes cache files older than max_age_hours.
        """
        current_time = datetime.now ().timestamp ()
        for filename in os.listdir (self.cache_dir):
            file_path = os.path.join (self.cache_dir, filename)
            if os.path.isfile (file_path):
                file_age = current_time - os.path.getmtime (file_path)
                if file_age > (max_age_hours * 3600):
                    try:
                        os.remove (file_path)
                        logging.info (f"Removed old cache file: {filename}")
                    except Exception as e:
                        logging.warning (f"Failed to remove cache file {filename}: {e}")


# ------------------------------------------------------------------------------------
# YouTube Download
# ------------------------------------------------------------------------------------
def download_youtube_video (url, output_filename):
    """
    Downloads a YouTube video using yt-dlp.
    """
    try:
        import yt_dlp

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_filename,
            'quiet': False,
            'nocheckcertificate': True,
        }

        with yt_dlp.YoutubeDL (ydl_opts) as ydl:
            logging.info (f"Downloading video from {url}")
            ydl.download ([url])

        if os.path.exists (output_filename):
            logging.info (f"Successfully downloaded video: {output_filename}")
            return output_filename
        else:
            logging.error ("Download completed but file not found")
            return None

    except Exception as e:
        logging.error (f"Failed to download video: {str (e)}")
        return None


# ------------------------------------------------------------------------------------
# Media Handling
# ------------------------------------------------------------------------------------
def extract_audio (video_path, audio_path):
    """Extracts audio from the given video path using ffmpeg."""
    try:
        if os.path.exists (audio_path):
            os.remove (audio_path)
        stream = ffmpeg.input (video_path)
        stream = ffmpeg.output (stream, audio_path)
        ffmpeg.run (stream)
        if os.path.exists (audio_path):
            return True
        else:
            logging.error ("Audio extraction failed: File not created")
            return False
    except Exception as e:
        logging.error (f"Failed to extract audio: {str (e)}")
        return False


def transcribe_audio(audio_path, model_size="medium"):
    """Transcribes the given audio file using OpenAI Whisper with GPU support."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(
            audio_path,
            fp16=(device == "cuda"),  # Use mixed precision on GPU
            language="en",
            task="transcribe"
        )
        return result["text"]
    except Exception as e:
        logging.error(f"Failed to transcribe audio: {str(e)}")
        return None

# ------------------------------------------------------------------------------------
# Transcript Utilities
# ------------------------------------------------------------------------------------
def split_transcript_into_chunks (transcript, chunk_size=CHUNK_SIZE):
    """
    Splits the transcript into multiple chunks for more efficient processing.
    """
    words = transcript.split ()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append (word)
        current_length += len (word)

        if current_length >= chunk_size:
            chunks.append (' '.join (current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append (' '.join (current_chunk))
    return chunks


# ------------------------------------------------------------------------------------
# PDF Creation
# ------------------------------------------------------------------------------------
def create_pdf (notes, keyframes, output_path):
    """
    Creates a PDF (lecture notes) that includes text and extracted keyframes.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate (output_path, pagesize=letter)
    styles = getSampleStyleSheet ()
    story = []

    # Add the summarized lecture notes
    for line in notes.split ('\n'):
        if line.strip ():
            story.append (Paragraph (line, styles['Normal']))
            story.append (Spacer (1, 12))

    temp_dir = os.path.join (CACHE_DIR, "keyframes")
    os.makedirs (temp_dir, exist_ok=True)

    for idx, (_, frame) in enumerate (keyframes):
        img_path = os.path.join (temp_dir, f'keyframe_{idx}.jpg')
        cv2.imwrite (img_path, frame)
        story.append (Image (img_path))
        story.append (Spacer (1, 12))

    doc.build (story)
    return True


# ------------------------------------------------------------------------------------
# Quality Metrics
# ------------------------------------------------------------------------------------
class QualityMetrics:
    """
    Evaluates the generated summaries based on length, coherence, sentiment, and redundancy.
    """

    def __init__ (self):
        self.sentiment_analyzer = None
        self.coherence_threshold = 0.7
        self.minimum_length = 100
        self.maximum_redundancy = 0.3

    def _init_sentiment_analyzer (self):
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = pipeline ('sentiment-analysis', device='cpu')

    def check_summary_quality (self, summary):
        """
        Returns an overall quality score (0 to 1) and a dictionary of individual metrics.
        """
        quality_scores = {
            'length': self._check_length (summary),
            'coherence': self._check_coherence (summary),
            'sentiment': self._check_sentiment (summary),
            'redundancy': self._check_redundancy (summary)
        }
        overall_score = sum (quality_scores.values ()) / len (quality_scores)
        return overall_score, quality_scores

    def _check_length (self, summary):
        words = len (summary.split ())
        if words < self.minimum_length:
            return words / self.minimum_length
        return 1.0

    def _check_coherence (self, summary):
        """
        Rough measure of coherence by checking how consecutive sentences overlap in words.
        """
        sentences = summary.split ('.')
        if len (sentences) < 2:
            return 0.0
        coherence_scores = []
        for i in range (len (sentences) - 1):
            current = set (sentences[i].lower ().split ())
            next_sent = set (sentences[i + 1].lower ().split ())
            overlap = len (current.intersection (next_sent))
            score = overlap / (len (current.union (next_sent)) + 1)
            coherence_scores.append (score)
        return sum (coherence_scores) / len (coherence_scores)

    def _check_sentiment (self, summary):
        """
        Analyzes sentiment using a simple huggingface pipeline.
        """
        try:
            self._init_sentiment_analyzer ()
            # Limit input length for pipeline demonstration
            result = self.sentiment_analyzer (summary[:512])[0]
            return result['score']
        except Exception as e:
            logging.warning (f"Sentiment analysis failed: {e}")
            return 0.5

    def _check_redundancy (self, summary):
        """
        Checks how many unique sentences exist to estimate redundancy.
        """
        sentences = summary.split ('.')
        unique_content = set (' '.join (s.lower ().split ()) for s in sentences)
        redundancy_score = len (unique_content) / (len (sentences) + 1)
        return min (1.0, redundancy_score / (1 - self.maximum_redundancy))


# ------------------------------------------------------------------------------------
# Keyframe Extraction
# ------------------------------------------------------------------------------------
class EnhancedKeyframeExtractor:
    """
    Uses feature extraction, KMeans clustering, and GLCM to select representative frames.
    """

    def __init__ (self, min_scene_length=MIN_SCENE_LENGTH):
        self.min_scene_length = min_scene_length

    def extract_keyframes (self, video_path, n_frames=10):
        """
        Extracts representative keyframes from a video by clustering feature vectors.
        """
        cap = cv2.VideoCapture (video_path)
        total_frames = int (cap.get (cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max (total_frames // (n_frames * 15), 2)

        frames = []
        features = []

        with tqdm (total=total_frames // sample_interval) as pbar:
            for frame_idx in range (0, total_frames, sample_interval):
                cap.set (cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read ()
                if not ret:
                    continue

                frame_features = self.extract_features (frame)
                features.append (frame_features)
                frames.append ((frame_idx, frame))
                pbar.update (1)

        # Cluster the frames and find frames closest to each cluster center
        kmeans = KMeans (n_clusters=n_frames, random_state=42)
        feature_matrix = np.array (features)
        clusters = kmeans.fit_predict (feature_matrix)

        selected_frames = []
        for i in range (n_frames):
            cluster_frames = [
                (idx, frm) for (idx, frm), cluster in zip (frames, clusters) if cluster == i
            ]
            if cluster_frames:
                center_features = kmeans.cluster_centers_[i]
                distances = [
                    np.linalg.norm (feature_matrix[frames.index ((idx, frm))] - center_features)
                    for idx, frm in cluster_frames
                ]
                selected_frames.append (cluster_frames[np.argmin (distances)])

        selected_frames.sort (key=lambda x: x[0])
        cap.release ()
        return selected_frames

    def extract_features (self, frame):
        """
        Extract\s a combination of statistical features, edges, histograms, and GLCM features.
        """
        gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)
        features = []

        try:
            if cv2.cuda.getCudaEnabledDeviceCount () > 0:
                gray_gpu = cv2.cuda_GpuMat ()
                gray_gpu.upload (gray)
                edges_gpu = cv2.cuda.createCannyEdgeDetector (100, 200)
                edges = edges_gpu.detect (gray_gpu).download ()
            else:
                edges = cv2.Canny (gray, 100, 200)
        except Exception as e:
            logging.warning (f"Failed to use CUDA for edge detection: {e}")
            edges = cv2.Canny (gray, 100, 200)

        # Histogram features for each channel of HSV
        # Histogram features for each channel of HSV
        for channel in range (3):
            hist = cv2.calcHist ([hsv], [channel], None, [256], [0, 256])
            features.extend (hist.flatten ())
        # GLCM features
        glcm = self.calculate_glcm (gray)
        features.extend ([
            self.glcm_contrast (glcm),
            self.glcm_homogeneity (glcm),
            self.glcm_energy (glcm)
        ])

        return np.array (features)

    def calculate_glcm (self, image, distance=1, angle=0):
        """
        Calculates a basic GLCM for a given grayscale image.
        """
        glcm = np.zeros ((256, 256))
        rows, cols = image.shape

        for i in range (rows - distance):
            for j in range (cols - distance):
                i2 = i + distance
                j2 = j + distance
                glcm[image[i, j], image[i2, j2]] += 1

        return glcm / glcm.sum ()

    def glcm_contrast (self, glcm):
        rows, cols = glcm.shape
        contrast = 0
        for i in range (rows):
            for j in range (cols):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast

    def glcm_homogeneity (self, glcm):
        rows, cols = glcm.shape
        homogeneity = 0
        for i in range (rows):
            for j in range (cols):
                homogeneity += glcm[i, j] / (1 + abs (i - j))
        return homogeneity

    def glcm_energy (self, glcm):
        return np.sum (glcm ** 2)


# ------------------------------------------------------------------------------------
# GPT4All Processing
# ------------------------------------------------------------------------------------
def setup_model(model_name=DEFAULT_MODEL_NAME):
    """Loads (and downloads if necessary) the GPT4All model into GPU memory if available."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpt = GPT4All(
            model_name=model_name,
            model_path=MODEL_DIR,
            device=device,  # Use GPU if available
            allow_download=True,
            n_threads=mp.cpu_count()  # Use all CPU cores for preparation
        )
        if device == "cuda":
            gpt.limit_memory (0.8)  # Limit to 80% of GPU memory
        logging.info(f"Successfully loaded/downloaded model: {model_name}")
        return gpt, os.path.join(MODEL_DIR, model_name)
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        return None, None


def process_chunk_parallel (chunk_data):
    """Worker function for parallel processing of transcript chunks with GPT4All."""
    (chunk, chunk_idx, gpt, keyframe_references) = chunk_data
    prompt = (
        "As an expert in creating academic lecture notes, your task is to:\n"
        "1. Summarize the following lecture transcript segment\n"
        "2. Create clear, structured notes using markdown formatting\n"
        "3. Reference relevant keyframes when they align with the content\n"
        "4. Maintain academic language and technical accuracy\n\n"
        f"Available Keyframes:\n{keyframe_references}\n\n"
        f"Transcript Chunk {chunk_idx + 1}:\n{chunk}\n\n"
        "Provide your summary in markdown format:\n"
    )

    try:
        logging.info (f"Starting generation for chunk {chunk_idx}")
        response = gpt.generate(
            prompt,
            max_tokens=512,  # Reduce token count to fit memory limits
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            stop_sequence=["###", "```"],
            batch_size=4  # Smaller batches for GPU efficiency
        )

        if response:
            logging.info ("Generation successful")
            metrics = QualityMetrics ()
            quality_score, detailed_metrics = metrics.check_summary_quality (response)

            return {
                "summary": response,
                "chunk_idx": chunk_idx,
                "quality_score": quality_score,
                "quality_metrics": detailed_metrics
            }
        else:
            logging.error ("Generation returned empty response")
            return None

    except Exception as e:
        logging.error (f"Error in process_chunk_parallel: {str (e)}", exc_info=True)
        return None

    metrics = QualityMetrics ()
    quality_score, detailed_metrics = metrics.check_summary_quality (summary)

    return {
        "summary": summary,
        "chunk_idx": chunk_idx,
        "quality_score": quality_score,
        "quality_metrics": detailed_metrics
    }


def summarize_chunks_with_visuals (transcript_chunks, keyframes, num_workers=1):
    """
    Utilizes GPT4All to summarize multiple chunks of transcript text,
    references keyframes, and combines results into final notes.
    """
    print ("Summarizing transcript in chunks...")

    cache_manager = CacheManager ()
    cache_key = cache_manager.get_cache_key (transcript_chunks[0])
    cached_result = cache_manager.get_cached_data (cache_key)
    if cached_result:
        return cached_result

    gpt, _ = setup_model ()
    if not gpt:
        logging.error ("Failed to setup model. Exiting.")
        return None

    keyframe_references = "\n".join ([
        f"- Frame {i}: Time {frame_idx / 30:.2f}s"
        for i, (frame_idx, _) in enumerate (keyframes)
    ])

    # Process chunks and track which ones fail
    results = []
    failed_indices = []
    for i, chunk in enumerate (transcript_chunks):
        try:
            result = process_chunk_parallel ((chunk, i, gpt, keyframe_references))
            if result:
                results.append (result)
            else:
                failed_indices.append (i)
                logging.warning (f"Failed to process chunk {i}")
        except Exception as e:
            failed_indices.append (i)
            logging.warning (f"Error processing chunk {i}: {e}")
            continue

    # Retry failed chunks
    for i in failed_indices:
        try:
            result = process_chunk_parallel ((transcript_chunks[i], i, gpt, keyframe_references))
            if result:
                results.append (result)
        except Exception as e:
            logging.warning (f"Failed to process chunk {i} on retry: {e}")

    if not results:
        return None

    # Combine results
    quality_threshold = 0.7
    good_summaries = [
        (r['summary'], r['chunk_idx'])
        for r in results if r['quality_score'] >= quality_threshold
    ]
    if not good_summaries:
        return None

    combined_prompt = (
            "Create a comprehensive set of lecture notes from these summaries. "
            "Use markdown formatting for headers and emphasis. Maintain references "
            "to relevant frames.\n\n" +
            "\n\n".join ([f"Section {i + 1}:\n{summary}" for summary, i in good_summaries]) +
            "\n\nFinal Notes:\n"
    )

    try:
        # Try with minimal parameters
        final_notes = gpt.generate (combined_prompt)

        if not final_notes:
            # If that doesn't work, try with basic configuration
            final_notes = gpt.generate (
                combined_prompt,
                max_tokens=256,
                top_k=40,
                top_p=0.4
            )
    except Exception as e:
        logging.error (f"Generation failed: {str (e)}")
        return None

    cache_manager.cache_data (cache_key, (final_notes, keyframes))
    return final_notes, keyframes

    # Combine results
    quality_threshold = 0.7
    good_summaries = [
        (r['summary'], r['chunk_idx'])
        for r in results if r['quality_score'] >= quality_threshold
    ]
    if not good_summaries:
        return None

    combined_prompt = (
            "Create a comprehensive set of lecture notes from these summaries. "
            "Use markdown formatting for headers and emphasis. Maintain references "
            "to relevant frames.\n\n" +
            "\n\n".join ([f"Section {i + 1}:\n{summary}" for summary, i in good_summaries]) +
            "\n\nFinal Notes:\n"
    )

    try:
        # Try with minimal parameters
        final_notes = gpt.generate (combined_prompt)

        if not final_notes:
            # If that doesn't work, try with basic configuration
            final_notes = gpt.generate (
                combined_prompt,
                max_tokens=256,
                top_k=40,
                top_p=0.4
            )
    except Exception as e:
        logging.error (f"Generation failed: {str (e)}")
        return None

    cache_manager.cache_data (cache_key, (final_notes, keyframes))
    return final_notes, keyframes


# ------------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------------
def main ():
    """
    Main workflow:
    1. Prompt user for YouTube URL and download video.
    2. Extract keyframes.
    3. Extract and transcribe audio.
    4. Split transcript and summarize.
    5. Create a PDF with notes and keyframes.
    6. Cleanup temporary files and old cache.
    """
    cache_manager = CacheManager ()

    try:
        youtube_url = input ("Enter the YouTube URL for the lecture video: ")
        video_cache_key = cache_manager.get_cache_key (youtube_url)
        video_path = cache_manager.get_cached_data (video_cache_key)

        # Download if not cached
        if not video_path:
            video_path = download_youtube_video (youtube_url, output_filename="lecture.mp4")
            if video_path:
                cache_manager.cache_data (video_cache_key, video_path)

        if not video_path:
            logging.error ("Download failed. Exiting.")
            return

        # Keyframe extraction
        keyframe_extractor = EnhancedKeyframeExtractor ()
        keyframes = keyframe_extractor.extract_keyframes (video_path)
        if not keyframes:
            logging.error ("Keyframe extraction failed. Exiting.")
            return

        # Audio extraction
        audio_path = TEMP_AUDIO_FILE
        if not extract_audio (video_path, audio_path):
            logging.error ("Audio extraction failed. Exiting.")
            return

        # Transcription
        transcript = transcribe_audio (audio_path, model_size="medium")
        if not transcript:
            logging.error ("Transcription failed. Exiting.")
            return

        # Split transcript and summarize
        transcript_chunks = split_transcript_into_chunks (transcript)
        logging.info (f"Transcript split into {len (transcript_chunks)} chunks.")

        summary_result = summarize_chunks_with_visuals (
            transcript_chunks,
            keyframes,
            num_workers=min (NUM_WORKERS, mp.cpu_count ())
        )
        if summary_result is None:
            logging.error ("Failed to generate summary. Exiting.")
            return

        final_notes, keyframes = summary_result

        # Create final PDF
        pdf_path = OUTPUT_PDF
        create_pdf (final_notes, keyframes, pdf_path)

        # Cleanup temporary audio
        if os.path.exists (audio_path):
            try:
                os.remove (audio_path)
                logging.info (f"Cleaned up temporary audio file: {audio_path}")
            except Exception as e:
                logging.warning (f"Failed to remove {audio_path}: {e}")

        # Attempt to clean up keyframe files if they exist
        for idx, (_, frame) in enumerate (keyframes):
            img_path = f'keyframe_{idx}.jpg'
            if os.path.exists (img_path):
                try:
                    os.remove (img_path)
                    logging.info (f"Cleaned up keyframe: {img_path}")
                except Exception as e:
                    logging.warning (f"Failed to clean up keyframe {img_path}: {e}")

        # Clean old cache entries
        try:
            cache_manager.cleanup_old_cache (max_age_hours=72)
            logging.info ("Cleaned up old cache entries")
        except Exception as e:
            logging.warning (f"Cache cleanup failed: {e}")

        logging.info ("Process completed successfully")
        print (f"\nLecture notes have been generated and saved to: {pdf_path}")
        print ("You can find the processing log in:", LOG_FILE)

    except KeyboardInterrupt:
        logging.warning ("Process interrupted by user")
        print ("\nProcess interrupted. Cleaning up...")
        # Attempt minimal cleanup
        if os.path.exists (TEMP_AUDIO_FILE):
            try:
                os.remove (TEMP_AUDIO_FILE)
            except:
                pass
    except Exception as e:
        logging.error (f"Unexpected error: {str (e)}", exc_info=True)
        print ("\nAn error occurred. Check video_processor.log for details.")
    finally:
        # Ensure any remaining temporary directories are removed
        temp_dir = "./keyframes"
        if os.path.exists (temp_dir):
            try:
                import shutil
                shutil.rmtree (temp_dir)
                logging.info ("Cleaned up temporary directory")
            except Exception as e:
                logging.warning (f"Failed to clean up temporary directory: {e}")


if __name__ == "__main__":
    try:
        main ()
    except Exception as e:
        logging.error (f"Fatal error: {str (e)}", exc_info=True)
        print ("\nA fatal error occurred. Please check video_processor.log for details.")