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
            temperature=0.3
        )
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="history")
        

    def generate_answer(self, question: str, context: list[str]) -> str:
        history = self.memory.load_memory_variables({})["history"]
        temp = """You are a helpful and friendly AI assistant.

        ### Instructions:
        1. Conversational & Personal Inputs:
            - For greetings, casual chat, or questions about the user (e.g., their name), use the Chat History to answer naturally.

        2. Technical Questions:
            - Use the provided Context AND the Chat History to answer questions accurately.
            - If the answer was mentioned previously in the Chat History (e.g., facts shared by the user), you are allowed to use it!
            - If a question cannot be answered using either the Context OR the Chat History, explicitly state that you do not know.
            - STRICT RULE: DO NOT use your general knowledge or outside information to answer questions. If the fact is not explicitly in the Context or History, you must stop at "I do not know."

        3. Tone & Length:
            - Structure your answers to always be 3 bullet points.

        ## Chat History:
        {history}

        ## Context:
        {context}

        ## User Question:
        {Question}
        """
        
        template = PromptTemplate.from_template(temp)
        prompt = template.format(context="\n".join(context),history=history ,Question=question)
        
        response = self.model.invoke(prompt).content
        self.memory.save_context({"input": question}, {"output": response})
        return response
