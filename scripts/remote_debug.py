import os

from dotenv import load_dotenv

load_dotenv()

import dataiku
from dataiku.llm.python import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage
from surf_planner.agent.build import AgentBuilder
from surf_planner.config import ProjectSettings

dataiku.set_remote_dss("http://localhost:11000/", os.environ.get("DKU_API_KEY"))


class MyLLM(BaseLLM):

    def __init__(self):
        settings = ProjectSettings()
        client = dataiku.api_client()
        builder = AgentBuilder(client, settings)
        self.agent = builder.build()

    def process(self, query, settings, trace):

        langchain_messages = []
        for msg in query["messages"]:
            if msg.get("role") == "assistant":
                langchain_messages.append(AIMessage(**msg))
            else:
                langchain_messages.append(HumanMessage(**msg))

        inputs = {"messages": langchain_messages}
        final_state = self.agent.invoke(inputs)
        final_answer_message = final_state["messages"][-1]
        return {"text": final_answer_message.content}


my_llm = MyLLM()
prompt = """ I'd like to plan a surf trip to Bayonne. I am departing from Paris. 
I'd like to go on the 16 December 2025 and return on the 20 November 2025. """
print(my_llm.process({"messages": [{"content": prompt, "role": "user"}]},None, None))
