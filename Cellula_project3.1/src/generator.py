import os
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://openrouter.ai/api/v1',
            model='openrouter/free',
            temperature=0.3,
            streaming=True
        )
        
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="history")
        
        temp = """You are a helpful assistant. Use the context and history to answer the query.
        
        History:
        {history}
        
        Context:
        {context}
        
        Query:
        {input}
        """
        self.generate_prompt = PromptTemplate(
            input_variables=["history", "context", "input"], 
            template=temp
        )

    def generate_answer(self, query: str, context_chunks: list[str]):
        history = self.memory.load_memory_variables({})["history"]
        combined_context = "\n\n".join(context_chunks)
        
        prompt_str = self.generate_prompt.format(
            history=history, 
            context=combined_context, 
            input=query
        )
        
        return self.model.stream(prompt_str)

    def save_to_memory(self, query: str, final_response: str):
        self.memory.save_context({"input": query}, {"output": final_response})