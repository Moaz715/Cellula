import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class Evaluator:
    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv('GROQ_KEY'),
            temperature=0.0,
            max_retries=2
        )
        
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="history")
        
        temp = """You are an expert relevance evaluator.
        STRICT RULES:
        - Check the relevance and accuracy of the generated answer against the context and user query.
        - If satisfactory, output EXACTLY "STATUS: PASS".
        - If not satisfactory, output "STATUS: FAIL" followed by feedback and improvements as bullet points.
        
        History:
        {history}
        
        Context:
        {context}
        
        User Query:
        {input}
        
        Generator Answer:
        {answer}
        """
        self.generate_prompt = PromptTemplate(
            input_variables=["history", "context", "input", "answer"], 
            template=temp
        )

    def evaluate_answer(self, query: str, answer: str, context_chunks: list[str]):
        history = self.memory.load_memory_variables({})["history"]
        combined_context = "\n\n".join(context_chunks)
        
        prompt_str = self.generate_prompt.format(
            history=history,
            input=query,
            answer=answer,
            context=combined_context,
        )
        response = self.model.invoke(prompt_str)
        verdict_text = response.content or ""
                
        print(f"Evaluator Verdict:\n{verdict_text}")
        return verdict_text
        

    def save_to_memory(self, query: str, final_response: str):
        self.memory.save_context({"input": query}, {"output": final_response})