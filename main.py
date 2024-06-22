from Podcast_Summarizer_AI_Agent import PodSumCrew

def run():
    youtube_url = input("Please enter a Youtube Podcast URL: "),
    slack_channel = input("Please enter the name of the Slack channel where you would like the agent to send the message: "),
    inputs = {"youtube_url": youtube_url, "slack_channel": slack_channel}
    result = PodSumCrew().crew().kickoff(inputs=inputs)

if __name__ == '__main__':
    run()
