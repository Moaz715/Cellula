from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.memory import ConversationBufferWindowMemory
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
            temperature=0.5
        )
        self.memory = ConversationBufferMemory(memory_key="history")
        

    def generate_answer(self, question: str, context: list[str]) -> str:
        history = self.memory.load_memory_variables({})["history"]
        temp = """
        you are assistant, Answer the next Question using provided context,
        If you don't know the answer, just say you don't know.
        answer should be within 200 words or lower only
        ## Chat History :
        {history}
        ## Context :
        {context}
        ## Question :
        {Question}
        """
        
        template = PromptTemplate.from_template(temp)
        prompt = template.format(context="\n".join(context),history=history ,Question=question)
        
        response = self.model.invoke(prompt).content
        self.memory.save_context({"input": question}, {"output": response})
        return response
