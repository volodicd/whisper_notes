# Video Lecture Processor

An advanced Python application that processes video lectures to automatically generate comprehensive lecture notes with visual aids. The tool uses AI-powered transcription, summarization, and keyframe extraction to create structured PDF documents from educational videos.

## Features

- YouTube video download and processing
- Automated audio extraction and transcription using OpenAI Whisper
- Intelligent keyframe extraction using computer vision techniques
- GPU-accelerated processing when available
- AI-powered summarization using GPT4All
- PDF generation with embedded keyframes and structured notes
- Smart caching system for improved performance
- Detailed logging and error handling

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for accelerated processing)
- FFmpeg

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- ffmpeg-python
- gpt4all
- opencv-python
- numpy
- whisper
- pillow
- tqdm
- scikit-learn
- transformers
- yt-dlp
- reportlab

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-lecture-processor.git
cd video-lecture-processor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

1. Run the main script:
```bash
python video_processor.py
```

2. Enter the YouTube URL when prompted.

3. The script will:
   - Download the video
   - Extract key frames
   - Transcribe the audio
   - Generate summarized notes
   - Create a PDF with notes and visual references

The final output will be saved as `lecture_notes.pdf` in the working directory.

## Configuration

Key settings can be adjusted in the script:

```python
CACHE_DIR = ".cache"
MODEL_DIR = "models"
CHUNK_SIZE = 1000
NUM_WORKERS = 2
MIN_SCENE_LENGTH = 30
```

- `CACHE_DIR`: Directory for cached files
- `MODEL_DIR`: Directory for AI models
- `CHUNK_SIZE`: Size of text chunks for processing
- `NUM_WORKERS`: Number of parallel workers
- `MIN_SCENE_LENGTH`: Minimum length of scene for keyframe extraction

## Advanced Features

### GPU Acceleration

The tool automatically detects and uses GPU acceleration when available for:
- Video processing
- Transcription
- AI model inference

### Caching System

Implements an intelligent caching system that:
- Stores processed results
- Reduces redundant processing
- Automatically cleans up old cache files
- Uses MD5 hashing for cache keys

### Quality Metrics

Includes a sophisticated quality metrics system that evaluates:
- Content coherence
- Sentiment analysis
- Redundancy detection
- Length optimization

## Troubleshooting

Common issues and solutions:

1. **GPU Memory Issues**
   - Adjust `torch.cuda.set_per_process_memory_fraction(0.9)` to a lower value
   - Try running in CPU-only mode by setting `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

2. **Audio Extraction Fails**
   - Ensure FFmpeg is properly installed
   - Check file permissions
   - Verify input video format

3. **Model Download Issues**
   - Check internet connection
   - Verify disk space
   - Ensure write permissions in `MODEL_DIR`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

- OpenAI Whisper for transcription
- GPT4All for text generation
- FFmpeg for media processing
- YT-DLP for video downloading
