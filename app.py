import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

st.set_page_config(
    page_title="Weather & News Assistant",
    page_icon="☼",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Instrument Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 50% 5%, #fff 0, #f8f9f7 48%, #f2f4f1 100%); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 850px; padding-top: 2.2rem; padding-bottom: 2rem; }
    .hero { text-align: center; padding: 4.2rem 0 1.6rem; }
    .hero-icon { display: inline-grid; place-items: center; width: 52px; height: 52px; border-radius: 17px; background: #e5efe7; color: #598166; font-size: 32px; }
    .eyebrow { color: #829185; font-size: 10px; font-weight: 600; letter-spacing: 1.8px; margin: 15px 0 10px; }
    .hero h1 { color: #26312a; font-size: clamp(28px, 5vw, 40px); letter-spacing: -1.2px; font-weight: 500; margin: 0; }
    .hero-copy { color: #7d867f; margin-top: 12px; font-size: 15px; }
    [data-testid="stChatMessage"] { border: 1px solid #e8ece7; border-radius: 16px; background: #fff; box-shadow: 0 3px 12px #25392908; padding: 1rem; }
    [data-testid="stChatInput"] { border-color: #dfe5de; border-radius: 17px; }
    .footnote { text-align:center; color:#a3aaa3; font-size:10px; margin-top:10px; }
    div.stButton > button { border-radius: 999px; border-color:#e5e9e4; background:#fff; color:#58635b; min-height:43px; white-space:normal; }
    div.stButton > button:hover { border-color:#b9cbbc; color:#35463a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@tool
def get_weather(city: str) -> str:
    """Get current weather information for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "OpenWeather API key is missing. Add OPENWEATHER_API_KEY to your .env file."
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        if response.status_code == 404:
            return f"City {city} was not found."
        if response.status_code == 401:
            return "Invalid OpenWeather API key."
        response.raise_for_status()
        data = response.json()
        return f"Weather in {city}: {data['main']['temp']}°C, {data['weather'][0]['description']}"
    except requests.RequestException:
        return "Weather service is unavailable. Please try again."
    except (KeyError, IndexError, ValueError):
        return "Unable to read the weather data."


@tool
def get_latest_news(query: str) -> str:
    """Search for recent news about a given topic."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key is missing. Add TAVILY_API_KEY to your .env file."
    try:
        tavily = TavilySearchResults(max_results=3, tavily_api_key=api_key)
        results = tavily.invoke(query)
        return str(results) if results else "No news results found."
    except Exception:
        return "News service is unavailable. Please try again."


SYSTEM_PROMPT = """
You are a helpful weather and news assistant.
1. Use get_weather for weather questions.
2. Use get_latest_news for recent news and events.
3. If a question needs both tools, use both tools.
4. Combine the results before giving your final answer.
5. If the destination or another essential detail is missing, ask the user a clarifying question.
6. Use previous messages to understand follow-up questions.
7. Never invent weather data or news.
8. If a tool fails, explain the problem clearly.
9. Give answers in simple English.
"""


@st.cache_resource
def make_agent():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file and restart Streamlit.")
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    return create_agent(
        llm,
        tools=[get_weather, get_latest_news],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ☼ Weather & News")
    st.caption("Your day, in context")
    if st.button("＋  New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Ask about current weather, recent news, or both.")

if not st.session_state.messages:
    st.markdown(
        '<div class="hero"><div class="hero-icon">☼</div><p class="eyebrow">YOUR DAY, IN CONTEXT</p>'
        '<h1>What would you like to know?</h1><p class="hero-copy">Ask about the weather, recent news, or both in one message.</p></div>',
        unsafe_allow_html=True,
    )
    prompt_columns = st.columns(3)
    suggestions = [
        "What's the weather in Tokyo right now?",
        "Latest news about electric cars",
        "I'm visiting Paris this weekend. Weather and news?",
    ]
    for column, suggestion in zip(prompt_columns, suggestions):
        with column:
            if st.button(suggestion, use_container_width=True, key=f"suggestion_{suggestion}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about a city or a news topic", max_chars=4000)
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent = make_agent()
            with st.spinner("Checking weather and headlines…"):
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
            answer = result["messages"][-1].content
            if not isinstance(answer, str):
                answer = str(answer)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as error:
            st.error(f"I couldn't complete that request. {error}")

st.markdown('<p class="footnote">AI can make mistakes. Check important details with the original source.</p>', unsafe_allow_html=True)
