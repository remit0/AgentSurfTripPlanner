import os

from dotenv import load_dotenv

load_dotenv()


import dataiku
from langchain_core.messages import HumanMessage
from surf_planner.agent.build import AgentBuilder
from surf_planner.config import ProjectSettings

dataiku.set_remote_dss("http://localhost:11000/", os.environ.get("DKU_API_KEY"))


if __name__ == '__main__':

    settings = ProjectSettings()
    client = dataiku.api_client()
    builder = AgentBuilder(client, settings)
    agent = builder.build()

    prompt = """ I'd like to plan a surf trip to Bayonne. I am departing from Paris. 
    I'd like to go on the 16 December 2025 and return on the 20 November 2025. """
    inputs = {"messages": [HumanMessage(content=prompt)]}
    final_state = agent.invoke(inputs)
    final_answer = final_state["messages"][-1].content

    print(final_answer)
