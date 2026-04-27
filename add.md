

## **Introduction**

In this lab, you will extend the Movie Industry Analytics Pipeline with **audio and video processing** capabilities. The Movie Database (TMDb) API provides not only images and metadata but is also commonly used alongside movie trailers, interviews, and promotional audio content available on platforms like YouTube. 

Processing these audio and video files trimming, converting, extracting audio tracks, extracting keyframes, and transcribing speech transforms rich unstructured media into searchable, structured data that powers recommendation engines, accessibility tools, and content analytics. This lab continues directly from the project structure established in Lab 6\. You will add audio format handling using **pydub**, video frame and audio extraction using moviepy, speech-to-text transcription using faster-whisper, and MongoDB storage of transcript data with timestamps linked back to source files.

## **Learning Objectives**

By the end of this lab you will be able to:

* Load audio files in various formats (WAV, MP3, FLAC, OGG) and inspect their properties  
* Trim, concatenate, adjust volume, and apply fade effects using pydub  
* Convert audio between formats with appropriate quality and bitrate settings  
* Load video files and extract duration, frame rate, resolution, and codec metadata  
* Extract the audio track from a video file as a standalone MP3  
* Extract keyframes at regular intervals and save individual frames as images  
* Transcribe short and long audio files using faster-whisper with word-level timestamps  
* Transcribe audio extracted from a video file  
* Handle long audio transcription with chunking and progress tracking  
* Store transcript segments in MongoDB with timestamps, confidence scores, and source metadata  
* Link transcripts to their original source audio/video files  
* Log all audio and video processing activities to the pipeline log

In the end you will have the following flow: 

RAW DATA → Loader → Processor → Transcriber → MongoDB → Analytics  
                                 ↘ Video → Frames \+ Audio

Your existing project already has: data/raw/, data/processed/, src/, storage/, logs/. This lab adds audio and video processing modules.  You need to add the following: 

- In data/raw add folders audio and video folders and there you will store downloaded audio and video files   
- In data/processed add audio, frames and transcripts folders, and these folders will be used for trimmed/normalized/converted audio, extracted keyframe images in png and JSON, TXT, SRT transcript files  
- In /src add audio\_processing folder and there add \_\_init\_\_.py, loader.py to load and inspect audio, processor.py to trim, fade, volume, concat, and convert audio files, and transcriber.py for faster-whisper transcription  
- In /src you also need to add video\_processing folder with the following files: \_\_init\_\_.py, loader.py to load & inspect video, extract audio, and frame\_extractor.py to save keyframes as images

## Part 1 \- Install Required Libraries and Tools

Activate your existing virtual environment and install the audio/video processing libraries:

1. pip install pydub moviepy faster-whisper  
   These libraries are used for:  
* **pydub** for high-level audio editing built on top of ffmpeg. Handles WAV, MP3, FLAC, OGG, AAC.  
* **moviepy** for video editing and frame extraction. Converts frames to NumPy arrays for Python processing.  
* **faster-whisper** for a fast reimplementation of OpenAI Whisper using CTranslate2. Up to 4x faster than the original with lower memory usage. No separate ffmpeg required.

2. brew install ffmpeg  
   Note: to Install ffmpeg on windows you need to download from [https://ffmpeg.org](https://ffmpeg.org) and add the bin/ folder to PATH

   Linux (Ubuntu / Debian) users can run this command: sudo apt install ffmpeg  
   To Verify if ffmpeg is installed correctly run: ffmpeg \-version

   Important**:** faster-whisper does NOT require a separate ffmpeg installation, it uses the PyAV library which bundles ffmpeg internally. However, pydub and moviepy still require ffmpeg on the PATH.

3. pip install opencv-python used for processing and analyzing images and videos, such as extracting frames, detecting objects, and performing real-time visual tasks.

4. If you want to verify all installations then run:  
   python \-c "from pydub import AudioSegment; print('pydub OK')"  
   python \-c "from moviepy import VideoFileClip; print('moviepy OK')"  
   python \-c "from faster\_whisper import WhisperModel; print('faster-whisper OK')"

5. Now update requirements.txt with by adding the following:   
   pydub\>=0.25.1  
   moviepy\>=2.0.0  
   faster-whisper\>=1.0.0  
   

In this lab, we use specific libraries for audio, video, and transcription tasks. While there are many alternatives available, the chosen tools provide the best balance between simplicity, performance, and practical usability in a data pipeline context.

## Part 2 \- Audio Fundamentals and Format Overview

Before writing any code, it is important to understand how audio is stored digitally and why different formats exist. This knowledge helps you make the right choices when building a processing pipeline. Sound is a continuous wave of air pressure. To store it digitally, two processes are applied:

* **Sampling**: the amplitude of the wave is measured at regular intervals. The number of measurements per second is the sample rate in Hertz (Hz). CD-quality audio uses 44,100 Hz. Video production uses 48,000 Hz.  
* **Quantization**: each measurement is stored as an integer. A 16-bit system provides 65,536 possible values. Higher bit depth means more precision and a wider dynamic range. Studio recordings use 24-bit.

| Format | Compression | Quality | File Size | Best Used For |
| :---: | :---: | :---: | :---: | :---: |
| WAV | None (uncompressed) | Highest | Very large | Editing, archival, processing pipelines |
| MP3 | Lossy | Good | Small | Universal distribution. 128–320 kbps. |
| FLAC | Lossless | Perfect | Medium (\~50% of WAV) | Audiophile archival, no quality loss |
| OGG | Lossy | Good | Small | Open-source. Games and the web. |
| AAC | Lossy | Better than MP3 | Small | Apple ecosystem, streaming services |

*Rule of thumb: use WAV for all processing, MP3/AAC for distribution, FLAC for archival.*

Every AudioSegment in pydub exposes these key properties:

| Property | pydub Attribute | Typical Value | Notes |
| :---: | :---: | :---: | :---: |
| Sample Rate | audio.frame\_rate | 44100 or 48000 Hz | How many samples per second |
| Channels | audio.channels | 1 (mono) or 2 (stereo) | Mono is sufficient for speech |
| Bit Depth | audio.sample\_width | 2 (16-bit) or 3 (24-bit) | Bytes per sample |
| Duration | len(audio) | milliseconds | Divide by 1000 for seconds |
| Bitrate | export parameter | 128k – 320k | Only for lossy compressed formats |

## Part 3 \- Download audio and video files

Currently, YouTube has completely blocked automatic downloads without a PO token. As a result, if your API includes video or audio files hosted on YouTube, you will need to download them manually. You can use [this link](https://v5.ytmp4.is/convert2) for this purpose. If your API does not include audio or video files, you may use some of the alternative resources provided for your assignment [here](https://drive.google.com/drive/folders/1cA4ZGOKJukDOsf-QBDN_bYMBVf444mVW?usp=drive_link). All audio files should be placed in the raw/audio directory, and all video files in the raw/video directory.

## [Part 4 \- Loading Audio and Inspecting Properties](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/audio_processing/loader.py)

In this part you should add loader.py and it will be used for handling loading audio in any format and printing a summary of its technical properties. To test if it works you need to go over all audio files in your folder and display all details for each audio file in data/raw/audio

Expected output will look something like:  
  filename             : sample.mp3  
  format               : MP3  
  duration\_sec         : 30.05  
  channels             : 2  
  channel\_type         : Stereo  
  frame\_rate\_hz        : 44100  
  bit\_depth            : 16  
  file\_size\_kb         : 481.3

## [Part 5 \-  Audio Operations: Trim, Concatenate, Volume, Fade](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/audio_processing/processor.py)

Create src/audio\_processing/processor.py. This module contains all manipulation functions. Because AudioSegment objects are immutable, every operation returns a new AudioSegment meaning that the original is never modified. After running script check data/processed/audio/ and there should be .wav,.mp3, and .flac files

## [Part 6 – Video Fundamentals and moviepy](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/video_processing/loader.py)

Video is the most complex form of unstructured data because it combines images (frames), audio, and time. Understanding the distinction between containers and codecs is essential before processing video files.

| Term | Examples | What It Is |
| :---: | :---: | :---: |
| Container | MP4, AVI, MOV, MKV, WebM | The packaging format that holds video, audio, and metadata tracks together. Think of it as a box. |
| Codec | H.264, H.265, VP9, AV1 | The algorithm that compresses and decompresses the actual video/audio data. Think of it as the recipe for packing the contents. |

The same codec can be stored in different containers. The same container can hold video encoded with different codecs. Video codecs, such as H.264 or VP9, have a significant impact on file size and video quality. While some codecs offer high compression (e.g., VP9), others like H.264 strike a balance between quality and encoding speed. Choosing the right codec can significantly reduce file size without sacrificing quality.

In the video\_processing module, create a loader.py file that handles loading video files and returns a VideoFileClip object. It should also include functionality for inspecting video properties and extracting audio tracks from video files, saving them as MP3 files.

Important: Always Close VideoFileClip because moviepy keeps the video file open in memory while a VideoFileClip object exists. Always call clip.close() when you are done, and use a try/finally block. Failing to close can cause file handle leaks, especially in batch processing loops. Video processing can be memory-intensive, especially when working with high FPS or high-resolution video (4K). It is recommended to use appropriate parameters for frame extraction to avoid overloading memory, such as reducing the resolution or selecting a lower FPS for larger video files.

***Memory warning: 1 minute of video at 30 fps \= 1,800 frames. A single 4K frame is \~24 MB uncompressed. Only extract the frames you actually need.***

## [Part 7 \-  Extracting Keyframes from Video](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/video_processing/frame_extractor.py)

A keyframe is a frame extracted at a regular time interval. Keyframes are used for video thumbnails, scene detection, and building machine-learning datasets. Create src/video\_processing/frame\_extractor.py and there create a function for saving a single frame from the video at timestamp t\_seconds, extracting one frame every interval\_seconds seconds.

In video analysis, there is a trade-off between FPS and resolution. Higher FPS provides smoother video but increases file size and computational requirements. On the other hand, increasing the video's resolution (e.g., 4K) enhances detail but also requires more memory and processing power.

## [Part 8 \- Speech-to-Text Transcription with faster-whisper](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/audio_processing/transcriber.py)

Transcription converts spoken language in audio and video into written text. Once transcribed, content becomes searchable, indexable, and analysable with standard NLP techniques. faster-whisper is a reimplementation of OpenAI Whisper that achieves up to 4x faster speed with lower memory usage. faster-whisper supports 99+ languages and auto-detects the language of the audio.

| Model | Parameters | Speed | Accuracy | Best For |
| :---: | :---: | :---: | :---: | :---: |
| tiny | 39M | Fastest | Low | Quick testing, real-time prototypes |
| base | 74M | Fast | Medium | Development, light use, clear speech |
| small | 244M | Moderate | Good | Good balance for most production tasks |
| medium | 769M | Slow | High | Production pipelines, high accuracy needed |
| large-v3 | 1.5B | Slowest | Best | Research, archival, maximum accuracy |

Now in audio\_processing create file transcriber.py and there you should implement functions for loading and managing a Whisper model, transcribing audio files into structured text with timestamps, and extracting detailed transcription segments. Additionally, include functions to save the transcription results in different formats such as JSON, plain text, and SRT subtitles. 

In this part you will use transcribe() and it is important to understand the transcribe() generator because it returns a lazy generator. Transcription happens incrementally as you iterate. This is memory-efficient for very long files, but means nothing runs until you loop or call list().

We use pydub, moviepy, and faster-whisper because they provide a simple, efficient, and practical solution for audio processing, video handling, and transcription within a data pipeline. These tools are lightweight, easy to integrate, and support fast, offline processing without relying on external APIs.

## [Part 9 \- Transcribing Short Audio and Audio from Video](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/audio_processing/test_transcribe.py)

In this test file, you should call the transcription function on a sample audio file, display the main transcription results such as detected language, duration, segments, and a preview of the full text, and optionally print word-level confidence scores. It should also test saving the generated transcript into JSON, TXT, and SRT formats. 

For video files, the workflow is: extract audio track first, then transcribe the audio. Transcribing audio is significantly faster and uses less memory than processing the full video

## [Part 10 \- Transcribing Longer Audio with Chunking and Progress Tracking](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/audio_processing/transcriber.py)

Long audio files present specific challenges: they consume significant RAM, risk timeouts, and provide no feedback on progress. The recommended approach is to split the audio into chunks, transcribe each chunk separately, and combine the results.

### 10.1 Why Chunk Long Audio?

| Challenge | Solution |
| :---: | :---: |
| Large files consume significant RAM | Split into 5-minute chunks with pydub slicing |
| Long transcription may timeout or crash | Process each chunk independently, save results |
| No progress indication by default | Use log\_progress=True or tqdm progress bar |
| Single point of failure: crash \= restart | Save each chunk's transcript before moving to next |
| No way to resume interrupted jobs | Check if chunk output file exists before processing |

Now add chunked\_transcribe to transcriber.py. We will add a function for transcribing long audio files by splitting them into smaller chunks, processing each chunk separately, and then combining the results into one final transcript. The function should also adjust timestamps to match the original audio timeline, support cached chunk transcripts, and save both per-chunk and combined transcript files as JSON. After that update test file to test new functions as well. 

## [Part 11 \-  Storing Transcripts in MongoDB with Timestamps](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/storage/mongo.py)

Transcripts must be stored with their source file path, timestamps, and model metadata so they can be searched and re-linked to their original audio/video. Each document in the transcripts collection represents one transcribed audio or video file.  Extend your existing storage/mongo.py file.

## [Part 12 – Pipeline Integration and Logging](https://github.com/unstructureddata2012/Movie-Industry-Analytics-Pipeline/blob/main/src/pipeline/run_pipeline.py)

Integrate all audio and video processing into your main run\_pipeline.py. All processing activities must be logged to the pipeline.log file established in previous labs.

## **Assignment Tasks:**

1. Load and inspect at least 3 audio files in different formats (WAV, MP3, FLAC or OGG) (5%)  
2. Trim audio to a specific segment and export the result and save it in data/processed/audio (5%)  
3. Concatenate at least 2 audio clips and export the combined file (5%)  
4. Adjust volume (+dB and \-dB) and apply fade-in/fade-out effects (5%)  
5. Convert at least one audio file between two different formats (10%)  
6. Load video files and print its properties (duration, fps, resolution) (add image to Read.me), and extract the audio track from a video file (10%)  
7. Extract keyframes from a video at a regular interval (e.g. every 5s or 10s) (10%)  
8. Transcribe a short audio file with faster-whisper; print segments with timestamps (add image in Read.me) (10%)  
9. Transcribe audio extracted from a video file (10%)  
10. Transcribe a longer audio file using chunking strategy (10%)  
11. Store transcripts in MongoDB with timestamps, language, model, and source path (10%)  
12. Logging: all audio/video operations logged to pipeline.log (10%)