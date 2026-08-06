# src/grader.py
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class Grade(BaseModel):
    is_relevant: bool = Field(
        description="Set to True if the document contains code or logic relevant to the question, otherwise False."
    )

class RelevanceGrader:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192", 
            temperature=0
        )
        
        self.structured_llm_grader = self.llm.with_structured_output(Grade)
        
        self.prompt = PromptTemplate(
            template="""You are a strict grader assessing the relevance of a retrieved code document to a user question.
            
            User Question: {question}
            
            Retrieved Document:
            {document}
            
            Evaluate if the document contains code, concepts, or logic that can help answer the question.
            """,
            input_variables=["question", "document"],
        )
        
        self.chain = self.prompt | self.structured_llm_grader

    def check_relevance(self, query: str, document_text: str) -> bool:
        try:
            result = self.chain.invoke({
                "question": query, 
                "document": document_text
            })
            return result.is_relevant
            
        except Exception as e:
            print(f"[Error] Relevance Grader failed: {e}")
            raise RuntimeError("The relevance check failed due to a network or API error. Please try again.")