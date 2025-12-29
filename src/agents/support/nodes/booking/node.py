# pip install -qU "langchain[anthropic]" to call the model
from langchain.agents import create_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agents.support.nodes.booking.tools import tools
from agents.support.nodes.booking.prompt import prompt_template

booking_node = create_agent(
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    # model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite"),
    tools=tools,
    system_prompt=prompt_template.format(),
)
|