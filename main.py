#main.py
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
from pathlib import Path

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
class PodcastCrew:
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
    def transcriber(self) -> Agent:
        return Agent(
            config=self.agents_config['transcriber'],
            tools=self.audio_tool,  # Uses audio transcriber tool for converting podcast audio to text.
            verbose=True,
            llm=self.llm_model,
            allow_delegation=False,
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'],
            tools=[],  # No specific tools required, relies on LLM for summarization.
            verbose=True,
            llm=self.llm_model,
            allow_delegation=False,
        )

    @agent
    def action_point_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['action_point_specialist'],
            tools=[],  # No specific tools required, relies on LLM for identifying actionable points.
            verbose=True,
            llm=self.llm_model,
            allow_delegation=False,
        )

    @agent
    def product_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['product_analyst'],
            tools=[],  # No specific tools required, relies on LLM for identifying product mentions.
            verbose=True,
            llm=self.llm_model,
            allow_delegation=False,
        )

    @agent
    def content_auditor(self) -> Agent:
        return Agent(
            config=self.agents_config['content_auditor'],
            tools=[],  # The captain may not use specific tools but oversees the crew.
            verbose=True,
            llm=self.llm_model,
            allow_delegation=False,  # The content auditor leads but does not delegate the final task review.
        )

    @task
    def transcription_task(self) -> Task:
        return Task(
            config=self.tasks_config['transcription_task'],
            tools=self.audio_tool,  # Uses audio transcriber tool for converting audio to text.
            agent=self.transcriber(),
        )

    @task
    def summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['summary_task'],
            tools=[],  # No specific tools needed.
            agent=self.summarizer(),
        )

    @task
    def actionable_insights_task(self) -> Task:
        return Task(
            config=self.tasks_config['actionable_insights_task'],
            tools=[],  # No specific tools needed.
            agent=self.action_point_specialist(),
        )

    @task
    def product_identification_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_identification_task'],
            tools=[],  # No specific tools needed.
            agent=self.product_analyst(),
        )

    @task
    def quality_audit_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_audit_task'],
            tools=[],  # No specific tools needed.
            context=[self.transcription_task(), 
                 self.summary_task(), 
                 self.actionable_insights_task(), 
                 self.product_identification_task()],
            agent=self.content_auditor(),
        )


    @crew
    def crew(self) -> Crew:
        """Creates a crew for the Podcast summarizer"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )

def load_css(file_name):
    """Load external CSS from a file and inject it into the app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def tooltip_html(text, tooltip_text):
    """Return the HTML for an 'i' icon with a tooltip."""
    return f"""
    <div class="textbox-container">
      <div class="tooltip">
        <div class="icon">i</div>
        <span class="tooltiptext">{tooltip_text}</span>
      </div>
    </div>
    """


def run():
    #Load CSS page formatting and format page
    st.set_page_config(page_title="Podcast Summary App", layout="centered")
    st.title("Podcast Summary App")
    load_css('style.css')

    podcast_url = st.text_input("Enter the YouTube URL of the podcast you want to analyze")

    if st.button("Process Podcast"):
        if podcast_url:
            #st.info("Analysis in progress. This may take a couple of minutes, please wait...")

            # Add a waiting spinner while the analysis process runs
            with st.spinner('Processing the podcast... this may take a few minutes.'):
                inputs = {'youtube_url': podcast_url}
                result = PodcastCrew().crew().kickoff(inputs=inputs)

            # Directly access the attributes of the result object
            raw_transcription = result.raw if hasattr(result, "raw") else "Transcription not available"
            token_usage = result.token_usage if hasattr(result, "token_usage") else None

            # Display the raw transcription in a text box
            st.text_area("Output:", value=raw_transcription, height=300)

            # Display the token usage
            if token_usage:
                token_info = (
                    f"Token Usage:\n"
                    f"Total Tokens: {token_usage.total_tokens}\n"
                    f"Prompt Tokens: {token_usage.prompt_tokens}\n"
                    f"Completion Tokens: {token_usage.completion_tokens}\n"
                    f"Successful Requests: {token_usage.successful_requests}"
                )
            else:
                token_info = "Token usage not available."

            # Add tooltip below the transcription text area with token usage
            st.markdown(tooltip_html("i", f"Summary:\n\n\n{token_info}"), unsafe_allow_html=True)

        else:
            st.write("Podcast URL is empty")

if __name__ == "__main__":
    run()
