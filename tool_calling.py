
from logging import config
import os
import requests

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

#create a tool

@tool
def get_weather(city: str) -> str:
    """Get current weather information for a given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "OpenWeather API key is missing."

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": api_key,
                "units": "metric"
            },
            timeout=10
        )

        if response.status_code == 404:
            return f"City {city} was not found."

        if response.status_code == 401:
            return "Invalid OpenWeather API key."

        response.raise_for_status()
        data = response.json()

        temperature = data["main"]["temp"]
        condition = data["weather"][0]["description"]

        return f"Weather in {city}: {temperature}°C, {condition}"

    except requests.RequestException:
        return "Weather service is unavailable. Please try again."
    except (KeyError, IndexError, ValueError):
        return "Unable to read the weather data."




@tool
def get_latest_news(query: str) -> str:
    """Search for recent news about a given topic."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "Tavily API key is missing."

    try:
        tavily = TavilySearchResults(
            max_results=3,
            tavily_api_key=api_key
        )

        results = tavily.invoke(query)

        if not results:
            return "No news results found."

        return str(results)

    except Exception:
        return "News service is unavailable. Please try again."

system_prompt = """
You are a helpful weather and news assistant.

1. Use get_weather for weather questions.
2. Use get_latest_news for recent news and events.
3. If a question needs both tools, use both tools.
4. Combine the results before giving your final answer.
5. If the destination or another essential detail is missing,
   ask the user a clarifying question.
6. Use previous messages to understand follow-up questions.
7. Never invent weather data or news.
8. If a tool fails, explain the problem clearly.
9. Give answers in simple English.
If an essential detail is missing, ask a short clarifying question.
Do not guess the destination or other important details.
Remember the user's answer and continue the task.
"""

llm = ChatGroq(model="openai/gpt-oss-120b")
memory = InMemorySaver()


agent = create_agent(
    llm,
    tools=[get_weather, get_latest_news],
    system_prompt=system_prompt,
    checkpointer=memory
)

if __name__ == "__main__":
    print("Agent is ready. Type your query (or 'exit' to quit):")
    config = {"configurable": {"thread_id": "user_1"}}



while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    print(result['messages'][-1].content)