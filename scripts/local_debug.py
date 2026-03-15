import os
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()


import dataiku
from langchain_core.messages import HumanMessage
from surf_planner.agent.build import AgentBuilder
from surf_planner.config import ProjectSettings

dataiku.set_remote_dss("http://localhost:11000/", os.environ.get("DKU_API_KEY"))


if __name__ == '__main__':

    settings = ProjectSettings()
    builder = AgentBuilder(settings)
    agent = builder.build()

    today = date.today()
    start_date = today + timedelta(days=1)  # Tomorrow
    end_date = start_date + timedelta(days=7)  # 7 days after start

    start_str = start_date.strftime("%d %B %Y")
    end_str = end_date.strftime("%d %B %Y")

    # --- F-String Prompt ---
    prompt = f""" I'd like to plan a surf trip to Bayonne. I am departing from Paris. 
    I'd like to go on the {start_str} and return on the {end_str}. """

    inputs = {"messages": [HumanMessage(content=prompt)]}
    final_state = agent.invoke(inputs)
    final_answer = final_state["messages"][-1].content

    print(final_answer)
