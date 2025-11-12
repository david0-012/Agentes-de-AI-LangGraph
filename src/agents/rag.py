from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
import random

# Inicializa modelo de lenguaje OpenAI GPT-5 Nano
llm = init_chat_model("openai:gpt-5-nano", temperature=0)

# Configura tool con archivo vectorial
file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_690bab2297a0819185301cbc12831ede"],
}

# Vincula la tool al modelo de lenguaje OpenAI GPT-5 Nano
llm = llm.bind_tools([file_search_tool])

# Define el estado del agente con campos adicionales
class State(MessagesState):
    customer_name: str
    phone: str
    my_age: str

from pydantic import BaseModel, Field

# Define el esquema de salida estructurada para extraer información de contacto
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name : str = Field(description="The name of the person")
    email : str = Field(description="The email address of the person")
    phone : str = Field(description="The phone number of the person")
    age : str = Field(description="The age of the person")

# Inicializa modelo de lenguaje Google Gemini-2.5 Flash Lite
llm_with_structured_output = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0)   
# Configura la salida estructurada del modelo con el esquema definido
llm_with_structured_output = llm_with_structured_output.with_structured_output(schema=ContactInfo)


def extractor(state: State):
    history = state["messages"] # Obtener el historial de mensajes
    customer_name = state.get("customer_name",None)
    new_state : State = {}
    if customer_name is None or len(history) >= 10:
        schema = llm_with_structured_output.invoke(history)
        new_state["customer_name"] = schema.name
        new_state["phone"] = schema.phone
        new_state["my_age"] = schema.age
    return new_state

def conversation(state: State):
    new_state : State = {}
    history = state["messages"] # Obtener el historial de mensajes
    last_message = history[-1] # Obtener el último mensaje del usuario
    customer_name = state.get("customer_name","John Doe") # Obtener el nombre del cliente o usar un valor predeterminado
    system_message = f"You are helpful assistant that helps a customer named {customer_name}."
    ai_message = llm.invoke([("system", system_message), ("user", last_message.text)]) # Invoca el modelo de lenguaje con el mensaje del usuario y el mensaje del sistema
    new_state["messages"] = [ai_message] # Actualiza el estado con la respuesta del modelo
    return new_state

from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("conversation", conversation)
builder.add_node("extractor", extractor)

builder.add_edge(START, "extractor")
builder.add_edge("extractor", "conversation")
builder.add_edge("conversation", END)

agent = builder.compile()