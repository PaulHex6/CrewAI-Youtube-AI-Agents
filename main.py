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
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from pathlib import Path
import time

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

        # Load audio transcriber tool
        self.audio_tool = [audio_transcriber_tool]

        # Configure model from OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gpt_llm = ChatOpenAI(
            temperature=0.3,
            openai_api_key=self.openai_api_key,
            model="gpt-4o-2024-08-06"               #for testing gpt-3.5-turbo, for production gpt-4o-2024-08-06
        )

        # Configure Perplexity Llama-3.1-Sonar using Perplexity's API
        self.perplexity_api_key = os.getenv("PPLX_API_KEY")
        self.pplx_llm = ChatPerplexity(
            model="llama-3.1-sonar-large-128k-online",
            temperature=0.3,
            pplx_api_key=self.perplexity_api_key
        )

    @agent
    def transcriber(self) -> Agent:
        return Agent(
            config=self.agents_config['transcriber'],
            tools=self.audio_tool,  # Uses audio transcriber tool for converting podcast audio to text.
            verbose=True,
            llm=self.gpt_llm,
            allow_delegation=False,
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'],
            tools=[],  # No specific tools required, relies on LLM for summarization.
            verbose=True,
            llm=self.gpt_llm,
            allow_delegation=False,
        )

    @agent
    def action_point_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['action_point_specialist'],
            tools=[],  # No specific tools required, relies on LLM for identifying actionable points.
            verbose=True,
            llm=self.gpt_llm,
            allow_delegation=False,
        )

    @agent
    def content_auditor(self) -> Agent:
        return Agent(
            config=self.agents_config['content_auditor'],
            tools=[],  # The captain may not use specific tools but oversees the crew.
            verbose=True,
            llm=self.gpt_llm,
            allow_delegation=False,  # The content auditor leads but does not delegate the final task review.
        )

    @agent
    def claims_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['claims_analyst'],
            tools=[],  # No tools being used
            verbose=True,
            llm=self.gpt_llm,  # Using the LLM model
            allow_delegation=False,
        )

    @agent
    def fact_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['fact_checker'],
            tools=[],  # No tools being used
            verbose=True,
            llm=self.pplx_llm,  # Using Perplexity with online access
            allow_delegation=False,
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
    def claims_identification_task(self) -> Task:
        return Task(
            config=self.tasks_config['claims_identification_task'],
            tools=[],  # No tools being used
            agent=self.claims_analyst(),  # Using the claims_analyst agent
        )

    @task
    def fact_checking_task(self) -> Task:
        return Task(
            config=self.tasks_config['fact_checking_task'],
            tools=[],  # No tools being used
            context=[self.claims_identification_task()],
            agent=self.fact_checker(),  # Using the fact_checker agent
        )

    @task
    def quality_audit_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_audit_task'],
            tools=[],  # No specific tools needed.
            context=[self.transcription_task(), 
                 self.summary_task(), 
                 self.actionable_insights_task(), 
                 self.fact_checking_task()],
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
    """Load CSS from the static folder in your project directory."""
    css_path = os.path.join(os.path.dirname(__file__), 'static', file_name)
    with open(css_path) as f:
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
    # Load CSS page formatting and format page
    st.set_page_config(page_title="Podcast Summary App", page_icon="▶️", layout="centered")
    load_css('style.css')
    st.title("⏯️ Video Fact Finder")

    # Sidebar for options and static text
    st.sidebar.title("How it Works")
    
    # Add static text for app introduction
    st.sidebar.markdown("""
        
    Welcome to ⏯️ **Video Fact Finder**, a tool designed to help you:
        
    - Summarize key points from YouTube videos.
    - Identify actionable insights and takeaways.
    - Fact-check claims made in the video for accuracy.

    Analyze your video content efficiently using AI-powered tools.

    *Note:* This tool analyzes only speech, so all results are based on transcription, not video.


    *Version 1.3.5*
    """)



    podcast_url = st.text_input("Enter the YouTube URL of the podcast you want to analyze")

    # Add Video Preview
    if podcast_url and "shorts" not in podcast_url:
        try:
            st.video(podcast_url)
        except Exception:
            pass  # Ignore any exception and don't display anything

    if st.button("Process Podcast"):
        if podcast_url:
            start_time = time.time()

            # Add a waiting spinner while the analysis process runs
            with st.spinner('Processing the podcast... this may take a few minutes.'):
                inputs = {'youtube_url': podcast_url}
                result = PodcastCrew().crew().kickoff(inputs=inputs)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Directly access the attributes of the result object
            raw_transcription = result.raw if hasattr(result, "raw") else "Transcription not available"
            token_usage = result.token_usage if hasattr(result, "token_usage") else None

            # Display the raw transcription with formatting and icons
            st.write(raw_transcription)

            # Display the token usage
            if token_usage:
                token_info = (
                    f"**Token Usage:**\n\n"
                    f"Total Tokens: {token_usage.total_tokens}\n"
                    f"Prompt Tokens: {token_usage.prompt_tokens}\n"
                    f"Completion Tokens: {token_usage.completion_tokens}\n"
                    f"Successful Requests: {token_usage.successful_requests}"
                )
            else:
                token_info = "Token usage not available."

            # Display the token info with a tooltip and the elapsed time
            st.markdown(tooltip_html("i", f"Summary:\n\n\n{token_info}\n\n"
                                        f"**Elapsed Time:** {elapsed_time:.2f} seconds"), 
                        unsafe_allow_html=True)

        else:
            st.write("Podcast URL is empty")

if __name__ == "__main__":
    run()
