# ▶️ Video Fact Finder

This project analyzes a YouTube video podcast and provides a summarized version of the content. Simply input the YouTube URL, and the CrewAI agents handles the rest.

#### Setup

Before running the application, add your API keys in a `.env` file.

#### Install the packages:
```bash
pip install -r requirements.txt
```

Before running this script, ensure FFmpeg is installed and added to your system's PATH.
Alternatively, you can specify the location of FFmpeg in the script using the 'ffmpeg_location' option.
For Windows users, download FFmpeg from https://ffmpeg.org/download.html and add the 'bin' folder to the PATH.

#### Time to spin up the application:
```bash
streamlit run main.py
```

### Acknowledgment:
This project is forked from the original work by Siddarth from Composio.
You can find the original repository and more examples on podcast summarization [here](https://github.com/ComposioHQ/composio/tree/master/python/examples/Podcast_summarizer_Agents).
