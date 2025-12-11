from dotenv import load_dotenv
import os
load_dotenv()

import dataiku
from dataiku.langchain import LangchainToDKUTracer
from dataiku.llm.python import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage
from surf_planner.build import AgentBuilder
from surf_planner.models import DKUChatLLM

default_calendar_id = "rosenthal.remi@gmail.com"
llm_model_id = "openai:OpenAI:gpt-4o"
dataiku.set_remote_dss("http://localhost:11000/", os.environ.get("DKU_API_KEY"))


class MyLLM(BaseLLM):

    def __init__(self):
        dku_client = dataiku.api_client()
        model = DKUChatLLM(llm_id=llm_model_id)
        agent_builder = AgentBuilder(
            client=dku_client,
            model=model,
            calendar_id=default_calendar_id
        )
        self.agent = agent_builder.build()

    def process(self, query, settings, trace):
        """
        Handles the full conversational turn.
        """
        tracer = LangchainToDKUTracer(dku_trace=trace)

        langchain_messages = []
        for msg in query["messages"]:
            if msg.get("role") == "assistant":
                langchain_messages.append(AIMessage(**msg))
            else:
                langchain_messages.append(HumanMessage(**msg))

        inputs = {
            "messages": langchain_messages,
        }

        final_state = self.agent.invoke(inputs)
        final_answer_message = final_state["messages"][-1]
        return {"text": final_answer_message.content}


my_llm = MyLLM()
print(my_llm.process({"messages": [{
    "content": "I'd like to plan a surf trip to Bayonne. I am departing from Paris. I'd like to go on the 2 November 2025 and return on the 10 November 2025.",
    "role": "user"}]},
    None, None))
