import os
import streamlit as st
import yaml
from dotenv import load_dotenv
import yt_dlp
import whisper
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import tool
from langchain.chat_models import ChatOpenAI
import time  # For simulating waiting time


load_dotenv()

@tool("Audio Transcribe Tool")
def audio_transcriber_tool(input_str: str) -> str:
    """
    Extracts audio and transcribes the audio from a YouTube video given its URL.

    Parameters:
    - input_str (str): A JSON string containing the URL of the YouTube video.

    Returns:
    str: The transcribed text from the YouTube audio.
    """
    import json
    try:
        # Parse the input string
        if input_str.strip().startswith('{'):
            inputs = json.loads(input_str)
            url = inputs.get('url') or inputs.get('input_str') or inputs.get('youtube_url')
            if url is None:
                raise ValueError("URL is required in the input JSON.")
        else:
            # Input is a plain URL
            url = input_str.strip()
            if not url:
                raise ValueError("Input URL is empty.")
    except json.JSONDecodeError as e:
        return f"Error parsing input JSON: {e}"
    except Exception as e:
        return f"Error processing input: {e}"

    # Proceed with downloading and transcription
    try:
        # Use yt-dlp to download the audio from the YouTube video
        ydl_opts = {
            'format': 'bestaudio/best',  # Download the best available audio
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',  # Extract as mp3
                'preferredquality': '192',  # Audio quality
            }],
            'outtmpl': 'audio_file.%(ext)s',  # Output filename with extension
            'quiet': True,  # Suppress console output
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])  # Download the audio from the URL

        # Find the downloaded audio file (assuming mp3 extension)
        audio_file = "audio_file.mp3"

        # Load the Whisper model
        whisper_model = whisper.load_model("small")

        # Transcribe the downloaded audio file
        result = whisper_model.transcribe(audio_file)

        # Clean up the audio file after transcription
        os.remove(audio_file)

        return result["text"]
    except Exception as e:
        return f"Error downloading or transcribing audio: {e}"

@CrewBase
class PodSumCrew:
    "Podcast summarizer Crew"

    def __init__(self):
        # Load configurations
        with open('config/agents.yaml', 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open('config/tasks.yaml', 'r') as f:
            self.tasks_config = yaml.safe_load(f)
        self.audio_tool = [audio_transcriber_tool]
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_model = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key,
            model="gpt-3.5-turbo"  # Specify model
        )

    @agent
    def summary_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Transcriber_summarizer'],
            tools=self.audio_tool,
            verbose=True,
            llm=self.llm_model,
            allow_delegation=False,
        )

    @task
    def generate_summary(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_podcast_task'],
            tools=self.audio_tool,
            agent=self.summary_agent(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates a crew for the Podcast summarizer"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )


def run():
    st.set_page_config(page_title="Podcast Summary App", layout="centered")
    st.title("Podcast Summary App")

    podcast_url = st.text_input("Enter the URL of the Podcast you want to summarize")

    if st.button("Summarize Podcast"):
        if podcast_url:
            st.info("Transcription in progress. This may take a couple of minutes, please wait...")

            # Add a waiting spinner while the transcription process runs
            with st.spinner('Processing the podcast... this may take a few minutes.'):
                inputs = {'youtube_url': podcast_url}
                result = PodSumCrew().crew().kickoff(inputs=inputs)

            # Directly access the attributes of the result object
            raw_transcription = result.raw if hasattr(result, "raw") else "Transcription not available"
            token_usage = result.token_usage if hasattr(result, "token_usage") else None

            # Display the raw transcription in a text box
            st.text_area("Transcription (Raw)", value=raw_transcription, height=300)

            # Display the token usage in a table if available
            if token_usage:
                st.write("**Token Usage**")
                token_data = {
                    "Metric": ["Total Tokens", "Prompt Tokens", "Completion Tokens", "Successful Requests"],
                    "Count": [
                        token_usage.total_tokens,
                        token_usage.prompt_tokens,
                        token_usage.completion_tokens,
                        token_usage.successful_requests
                    ]
                }
                st.table(token_data)  # Display the token data as a table

        else:
            st.write("Podcast URL is empty")

if __name__ == "__main__":
    run()
