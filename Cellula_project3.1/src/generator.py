import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.model = ChatGroq(
            api_key=os.getenv('GROQ_KEY'),
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_retries=2
        )
        
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="history")
        
        temp = """You are a strict data-retrieval assistant. 
        
        STRICT RULES: 
        1. Answer the query using ONLY the provided Context. 
        2. Keep your answer as short as possible. Use no more than 3 sentences unless the user explicitly asks for a list or table.
        3. Do NOT use external knowledge. 
        4. If the context lacks the information, output: "I cannot answer this based on the provided documents."
        
        History: {history}
        Context: {context}
        Evaluator Feedback: {feedback}
        Query: {input}
        """
        self.generate_prompt = PromptTemplate(
            input_variables=["history", "context", "feedback", "input"], 
            template=temp
        )

    def generate_answer(self, query: str, context_chunks: list[str], feedback: str = "None"):
        history = self.memory.load_memory_variables({})["history"]
        combined_context = "\n\n".join(context_chunks)
        
        prompt_str = self.generate_prompt.format(
            history=history, 
            context=combined_context, 
            feedback=feedback,
            input=query
        )
        response = self.model.invoke(prompt_str)
        answer_text = response.content or ""
                
        print(f"Generator Output:\n{answer_text}")
        return answer_text

    def save_to_memory(self, query: str, final_response: str):
        self.memory.save_context({"input": query}, {"output": final_response})