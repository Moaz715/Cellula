import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://openrouter.ai/api/v1',
            model='openrouter/free',
            temperature=0.0
        )
        template = """Analyze the user query and classify its intent into exactly one of these two categories:
        - 'EXPLAIN': If the user is asking to explain, debug, describe existing code/concepts or just pasted code with no text about it.
        - 'GENERATE': If the user is asking to write, generate, create, or modify code.

        Query: {query}
        Intent (Respond with ONLY 'EXPLAIN' or 'GENERATE'):"""
        self.prompt = PromptTemplate(input_variables=["query"], template=template)

    def classify(self, query: str) -> str:
        formatted = self.prompt.format(query=query)
        response = self.llm.invoke(formatted).content.strip().upper()
        return "EXPLAIN" if "EXPLAIN" in response else "GENERATE"