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
        
        temp = """You are a strict, merciless grading robot for a RAG system.
        
        STRICT RULES:
        1. Read the Context and the Generator Answer.
        2. The Generator Answer MUST NOT contain any facts, numbers, rules, or details that are not explicitly written in the Context. 
        3. If the Generator Answer includes ANY outside knowledge, even if it is factually correct in the real world, you MUST output "STATUS: FAIL" and list the hallucinated facts.
        4. If the answer is completely grounded ONLY in the Context, or if it correctly states the context doesn't have the answer, output EXACTLY "STATUS: PASS".
        
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