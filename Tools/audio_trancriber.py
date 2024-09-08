from crewai_tools import tool
import yt_dlp
import whisper

# @tool("Audio Transribe tool")
def audio_transcriber_tool(url):
    """
    Extracts audio and transcribes the audio from a YouTube video given its URL and summarizes it.

    Parameters:
    - url (str): The URL of the YouTube video from which audio will be extracted.

    Returns:
    str: A string containing:
        - The summarized version of the Transcribed Youtube URL
    """
    
    # Use yt-dlp to download the audio from the YouTube video
    ydl_opts = {
        'format': 'bestaudio/best',  # Download the best available audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # Extract as mp3
            'preferredquality': '192',  # Audio quality
        }],
        'outtmpl': 'audio_file'  # Output filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])  # Download the audio from the URL

    # Load the Whisper model
    whisper_model = whisper.load_model("small")
    
    # Transcribe the downloaded audio file
    result = whisper_model.transcribe("audio_file.mp3")
    
    return result["text"]