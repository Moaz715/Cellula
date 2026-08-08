# src/reformulator.py
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class QueryReformulator:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url='https://openrouter.ai/api/v1',
            model='openrouter/free',
            temperature=0.0
        )
        template = """You are a search query optimizer. Given the chat history and the user's follow-up, rewrite the follow-up to be a standalone search query.

CRITICAL RULES:
1. ONLY replace pronouns ("it", "that") or vague references ("that function") with the specific entity from the history.
2. Keep the query as SHORT and MINIMAL as possible. DO NOT add conversational filler like "How can I..." or "Write a python script that...".
3. If the query is already standalone, return it EXACTLY as written with zero changes.
4. DO NOT include prefixes, labels, or extra text like "Standalone Query:". Output ONLY the raw search query.

EXAMPLES:
History: User: "Write a code that adds two numbers"
Follow-up: "Now change that function to multiply instead"
Standalone Query: multiply two numbers

History: User: "Explain binary search"
Follow-up: "What is its time complexity?"
Standalone Query: binary search time complexity

ACTUAL CHAT HISTORY:
{history}

User Follow-up: {query}
Output ONLY the final query:"""
        self.prompt = PromptTemplate(input_variables=["history", "query"], template=template)

    def reformulate(self, query: str, messages: list) -> str:
        if not messages:
            print('StandAlone Query:', query)
            return query
            
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[-4:]])
        
        formatted = self.prompt.format(history=history_text, query=query)
        raw_output = self.llm.invoke(formatted).content.strip()
        
        cleaned = re.sub(r'^(Standalone Query|Standalone query|Query):\s*', '', raw_output, flags=re.IGNORECASE).strip()
        
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
            
        print('StandAlone Query:', cleaned)
        return cleaned