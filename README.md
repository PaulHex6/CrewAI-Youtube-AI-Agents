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

##### connect composio with slack and create an app
```bash
https://docs.composio.dev/apps/slack :- follow this documentation to connect slack and create an app.
```

##### Time to spin up the application
```bash
streamlit run main.py
```

# setup(with docker):-

```bash
download docker and spin up the docker daemon
docker build 
```

2) crewai :- https://www.crewai.com/