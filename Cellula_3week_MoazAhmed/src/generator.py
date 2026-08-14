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
        1. Base your code ONLY on the provided verified context chunks. If an 'Official Solution' is provided, you MUST use its logic exactly instead of inventing your own.
        2. ALWAYS wrap your final executable code in a standard markdown block: ```python ... ```
        3. The context includes 'Official Tests' for various chunks. You MUST append ONLY the specific tests that correspond to the function you are actually generating. Ignore tests for other functions.
        4. Add `print("All tests passed successfully!")` at the very end of the code block so the user knows the assertions succeeded.

        Verified Context (Prompt, Solution & Tests):
        {context}

        Previous Conversation:
        {history}

        Query:
        {input}
        """
        self.generate_prompt = PromptTemplate(input_variables=["history", "context", "input"], template=generate_temp)
        
        voice_temp = """You are an expert SQL database administrator. Your job is to translate spoken user commands into executable SQL queries.
        STRICT RULES:
        1. Output ONLY the executable SQL query. Do not include markdown code blocks (like ```sql), conversational text, or explanations.
        2. Read the "Database Schema" section below to find the correct table and column names. Never invent tables or columns.
        3. Base your query strictly on the user's spoken command. Do not add external instructions, filters, or logic not mentioned by the user.

        ### Database Schema:
        {schema}

        ### Spoken Command:
        {input}
        """
        self.voice_prompt = PromptTemplate(input_variables=['schema','input'], template=voice_temp)

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
    
    def generate_sql(self, query: str, schema: str):
        prompt_str = self.voice_prompt.format(
            schema=schema,
            input=query
        )
        res = self.model.invoke(prompt_str)
        raw_sql = res.content.strip()
        return raw_sql

    def save_to_memory(self, query: str, final_response: str):
        self.memory.save_context({"input": query}, {"output": final_response})