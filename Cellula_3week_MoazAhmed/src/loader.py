from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from datasets import load_dataset
class DataLoader:
    @staticmethod
    def load()->list[Document]:
        hf_data = load_dataset('openai/openai_humaneval', split='test')
        
        documents = []
        for row in hf_data:
            doc = Document(
                page_content=row['prompt'],
                metadata={
                    'task_id': row['task_id'],
                    'entry_point': row['entry_point'],
                    'solution': row['canonical_solution'],
                    'test_code': row['test']
                }
            )
            documents.append(doc)
        return documents
        
        