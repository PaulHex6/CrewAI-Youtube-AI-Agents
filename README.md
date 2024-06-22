# Podcast_summarizer_Agents
 Summarizing a whole podcast form youtube and send it as a slack message to a channel using crewai and composio.


# setup(without docker):-

##### clone the repository
```bash
 git clone https://github.com/siddartha-10/Podcast_summarizer_Agents.git
```

##### Install the packages
```bash
pip install -r requirements.txt
```

##### connecting composio with slack
```bash
https://docs.composio.dev/apps/slack
```

##### Time to spin up the application
```bash
streamlit run main.py
```

# setup(with docker):-

```bash
download docker and spin up the docker daemon
docker build . -t <app_name>
docker run -p 8501:8501 <app_ID>
```