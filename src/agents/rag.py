from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
import random

llm = init_chat_model("openai:gpt-5-nano", temperature=0)
file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_690bab2297a0819185301cbc12831ede"],
}

ll = llm.bind_tools([file_search_tool])
class State(MessagesState): # 
    customer_name: str
    my_age: int

def extractor(state: State):
    return {}

def conversation(state: State):
    new_state : State = {}
    if state.get("customer_name") is None:
        new_state ["customer_name"] = "John Doe"
    else:
        new_state["my_age"] = random.randint(18,65)
        
    history = state["messages"] # Obtener el historial de mensajes
    last_message = history[-1] # Obtener el último mensaje del usuario
    ai_message = llm.invoke(last_message.text) # Llama al modelo con el último mensaje del usuario
    new_state["messages"] = [ai_message] # Actualiza el estado con la respuesta del modelo
    return new_state

from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("conversation", conversation)
builder.add_node("extractor", extractor)

builder.add_edge(START, "conversation")
builder.add_edge("extractor", "conversation")
builder.add_edge("conversation", END)

agent = builder.compile()