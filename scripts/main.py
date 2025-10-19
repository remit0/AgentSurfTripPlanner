from dotenv import load_dotenv
import os
load_dotenv()


import dataiku
from langchain_core.messages import HumanMessage
from surf_planner.build import AgentBuilder
from surf_planner.models import DKUChatLLM

dataiku.set_remote_dss("http://localhost:11000/", os.environ.get("DKU_API_KEY"))
default_calendar_id = "rosenthal.remi@gmail.com"
llm_model_id = "openai:OpenAI:gpt-4o"
prompt = """Plan a surftrip to Limoges"""


if __name__ == '__main__':
    import dataiku
    client = dataiku.api_client()
    project = client.get_project("AGENTSURFTRIPPLANNER")
    dataset = project.get_dataset("conversations")
    print(dataset.get_as_core_dataset().get_dataframe())
    print(test)
    dku_client = dataiku.api_client()
    model = DKUChatLLM(llm_id=llm_model_id)
    agent_builder = AgentBuilder(
         client=dku_client,
         model=model,
         calendar_id=default_calendar_id
     )
    print(prout)
    agent = agent_builder.build()
    inputs = {"messages": [HumanMessage(content=prompt)]}
    final_state = agent.invoke(inputs)
    final_answer = final_state["messages"][-1].content
    print(final_answer)
