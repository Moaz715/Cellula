from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
class ResponseGenerator:
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://openrouter.ai/api/v1',
            model='openrouter/free',
            temperature=0.8
        )
        

    def generate_answer(self, question: str, context: list[str]) -> str:
        temp = """
        you are assistant, Answer the next Question using provided context,
        If you don't know the answer, just say you don't know.
        answer should be within 200 words or lower only
        ## context :
        {context}
        ## Question :
        {Question}
        """
        
        template = PromptTemplate.from_template(template)
        prompt = template.format(context="\n".join(context), Question = question)
        
        response = self.model.invoke(prompt).content
        return response
