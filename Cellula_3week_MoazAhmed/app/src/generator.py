# src/generator.py
import os
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ResponseGenerator:
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://openrouter.ai/api/v1',
            model='openrouter/free',
            temperature=0.3,
            streaming=True
        )
        
        
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="history")
        
        
        explain_temp = """You are a senior software engineer. Explain the following code or concept briefly and concisely.
        STRICT RULES: 
        1. Keep your explanation extremely short (maximum 3-4 sentences or a few short bullet points).
        2. Do NOT invent details. Do NOT output long essays.
        
        Previous Conversation:
        {history}
        
        Query:
        {input}
        """
        self.explain_prompt = PromptTemplate(input_variables=["history", "input"], template=explain_temp)
        
       
        generate_temp = """You are an expert Python coding assistant.
        STRICT RULES: 
        1. Base your code solution ONLY on the provided verified context chunks.
        2. ALWAYS wrap your final executable code in a standard markdown block: ```python ... ```
        3. The context includes 'Official Tests'. You MUST append these exact tests to the bottom of your code block.
        4. Add `print("All tests passed successfully!")` at the very end of the code block so the user knows the assertions succeeded.

        Verified Context (Code & Tests):
        {context}

        Previous Conversation:
        {history}

        Query:
        {input}
        """
        self.generate_prompt = PromptTemplate(input_variables=["history", "context", "input"], template=generate_temp)

    def explain_answer(self, query: str):
        history = self.memory.load_memory_variables({})["history"]
        prompt_str = self.explain_prompt.format(
            history=history, 
            input=query
        )
        response_stream = self.model.stream(prompt_str)
        return response_stream

    def generate_answer(self, query: str, context_chunks: list[str]):
        history = self.memory.load_memory_variables({})["history"]
        
        combined_context = "\n\n".join(context_chunks)
        
        prompt_str = self.generate_prompt.format(
            history=history, 
            context=combined_context, 
            input=query
        )
        
        response_stream = self.model.stream(prompt_str)
        
        return response_stream

    def save_to_memory(self, query: str, final_response: str):
        """
        Because we are streaming the output in Streamlit, we must manually 
        save the final compiled string to LangChain's memory after it finishes streaming.
        """
        self.memory.save_context({"input": query}, {"output": final_response})