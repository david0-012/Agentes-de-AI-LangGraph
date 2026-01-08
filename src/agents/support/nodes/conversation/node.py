from langchain.chat_models import init_chat_model

from agents.support.state import State
from agents.support.nodes.conversation.prompt import SYSTEM_PROMPT
from agents.support.nodes.conversation.tools import tools
from langchain_core.messages import AIMessage
# Inicializa modelo de lenguaje OpenAI GPT-5 Nano
llm = init_chat_model("openai:gpt-5-nano", temperature=0)
# Vincula la tool al modelo de lenguaje OpenAI GPT-5 Nano
llm = llm.bind_tools(tools)

def conversation(state: State):
    new_state : State = {}
    history = state["messages"] # Obtener el historial de mensajes
    last_message = history[-1] # Obtener el último mensaje del usuario
    customer_name = state.get("customer_name","John Doe") # Obtener el nombre del cliente o usar un valor predeterminado
    ai_message = llm.invoke([("system", SYSTEM_PROMPT), ("user", last_message.text)]) # Invoca el modelo de lenguaje con el PROMPT y el mensaje del usuario
    ai_message = AIMessage(content=ai_message.text)
    new_state["messages"] = [ai_message] # Actualiza el estado con la respuesta del modelo
    return new_state