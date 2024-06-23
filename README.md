# Podcast_summarizer_Agents
Summarizing a whole podcast form youtube and send it as a slack message to a channel using composio and crewai.
Simply input the YouTube podcast URL and your preferred Slack channel—The Crew handles the rest, summarizing the content and delivering it seamlessly.
 ###### The below images is how the crew executes
![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/siddartha/Desktop/github/Podcast_summarizer_Agents/images/test_run.png?version%3D1719100736141)

###### This is the slack that crew send
![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/siddartha/Desktop/github/Podcast_summarizer_Agents/images/Slack_message.png?version%3D1719100797641)

###### inputs for the above execution
```bash
youtube_url = https://www.youtube.com/watch?v=7T4-aEuGajI
slack_channel = "a-sumary-channel" (noticed a Typo).
```

# setup:-

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