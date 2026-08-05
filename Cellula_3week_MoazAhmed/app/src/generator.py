from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline
import os
load_dotenv()

class ResponseGenerator():
    def __init__(self):
        self.intentModel = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        self.labels = ["generate code", "explain code"]
        self.model = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://openrouter.ai/api/v1',
            model='openrouter/free',
            temperature=0.3
        )
        self.memory = ConversationBufferWindowMemory(k=5)
        temp = """You are a senior software engineer. Explain the following code clearly and concisely and in detail. Do not invent any details that are not present in the query.
        
        Previous Conversation:
        {history}
        
        Query:
        {input}
        """
        self.prompt = PromptTemplate(input_variables=["history", "input"], template=temp)
        self.chain = ConversationChain(
            llm=self.model,
            memory=self.memory,
            prompt=self.prompt
        )
        
    
    def classify_intent(self, query):
        res = self.intentModel(query, self.labels)
        return res['labels'][0]
    
    def explain(self, query):
        res = self.chain.predict(input=query)
        return res