from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from agents.support.state import State
from agents.support.nodes.extractor.prompt import SYSTEM_PROMPT

# Define el esquema de salida estructurada para extraer información de contacto
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name : str = Field(description="The name of the person")
    email : str = Field(description="The email address of the person")
    phone : str = Field(description="The phone number of the person")
    age : str = Field(description="The age of the person")

# Inicializa modelo de lenguaje Google Gemini-2.5 Flash Lite
# llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0)

# Inicializa modelo de lenguaje OpenAI GPT-4o Mini
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
   
# Configura la salida estructurada del modelo con el esquema definido
llm = llm.with_structured_output(schema=ContactInfo)


def extractor(state: State):
    history = state["messages"] # Obtener el historial de mensajes
    customer_name = state.get("customer_name",None)
    new_state : State = {}
    if customer_name is None or len(history) >= 10:
        schema = llm.invoke([("system", SYSTEM_PROMPT)] + history)
        new_state["customer_name"] = schema.name
        new_state["phone"] = schema.phone
        new_state["my_age"] = schema.age
    return new_state