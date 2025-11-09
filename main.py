from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient

# from langchain_classic import hub
# from langchain_classic.agents import AgentExecutor
# from langchain_classic.agents.react.agent import create_react_agent

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool to search over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Calling search tool for query : {query}")
#     # return "Chennai is a good city"
#     return tavily.search(query=query)

llm = ChatOllama(
    model="gemma3:270m"
)

tools = [TavilySearch()]

agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course-mpt!")
    response = agent.invoke({"messages":HumanMessage(content="Which is the best place in this world?")})
    print(response)


if __name__ == "__main__":
    main()
 