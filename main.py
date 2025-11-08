from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    # print("Hello from langchain-course-mpt!")

    information = """
    Pichai Sundararajan (born June 10, 1972), better known as Sundar Pichai (pronounced: /ˈsʊndɜːr pɪˈtʃeɪ/), is an Indian-American business executive.[3][4][5][6] He is the chief executive officer (CEO) of Alphabet Inc. and its subsidiary Google.[7]

    Pichai began his career as a materials engineer. Following a short stint at the management consulting firm McKinsey & Co., Pichai joined Google in 2004,[8] where he led the product management and innovation efforts for a suite of Google's client software products, including Google Chrome and ChromeOS, as well as being largely responsible for Google Drive. In addition, he went on to oversee the development of other applications such as Gmail and Google Maps.

    Pichai was selected to become the next CEO of Google on August 10, 2015, after previously being appointed chief product officer by then CEO Larry Page. On October 24, 2015, he stepped into the new position at the completion of the formation of Alphabet Inc., the new holding company for the Google company family. He was appointed to the Alphabet Board of Directors in 2017.[9] As of May 2025, his net worth is estimated at US$1.1 billion.[10]
    """

    summary_template = """
    you are the best summarizer in the world. You are an expert than no one can beat you in summarizing.Given the information about a person : {information}. I want you to create summary as aa first section and story as a second second section.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    llm = ChatOllama(
        model = "gemma3:270m",
        num_predict=512,
        temperature=0
    )

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information" : information})
    print(response.content)
    print(response.response_metadata["model"])


if __name__ == "__main__":
    main()
