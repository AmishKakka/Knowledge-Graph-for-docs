from LLMGraphTransformer import LLMGraphTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint
from time import time
import uuid


class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                            chunk_overlap=250,
                                                            length_function=len,
                                                            separators=["\n\n", "\n"])

    def get_doc_id(self):
        return str(uuid.uuid4())

    # def load_and_split(self, doc_path: str):
    #     print("Loading and splitting the document...")
    #     self.loader = PyMuPDFLoader(doc_path)
    #     doc = self.loader.load()
    #     loaded_docs = self.text_splitter.split_documents(doc)
    #     return loaded_docs

    def lazy_load_and_split(self, doc_path: str):
        print("Lazy loading and splitting the document...")
        self.loader = PyMuPDFLoader(doc_path)
        self.doc_id = self.get_doc_id()
        loaded_docs = []
        for doc in self.loader.lazy_load():
            loaded_docs.extend(self.text_splitter.split_documents([doc]))
        return loaded_docs

    def create_chunks(self, chunks: list):
        chunks_with_metadata = []
        for idx, c in enumerate(chunks):
            chunks_with_metadata.append({
              'text': c.page_content,
              'file_path': c.metadata['file_path'],
              'page': c.metadata['page'],
              'chunkId': self.doc_id+'_'+str(c.metadata['page'])+'_'+str(idx)
            })
        print(f"Created {len(chunks)} chunks with metadata...")
        return chunks_with_metadata

  
if __name__ == "__main__":
    doc_loader = DocumentLoader()

    # st = time()
    # document_splits = doc_loader.load_and_split()
    # et = time()
    # print("Total time taken: ", et-st)

    st = time()
    document_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")
    et = time()
    print("Total time taken to load, split, chunk document : ", et-st)
    print("\nTotal document splits: ", len(document_splits))
  
    chunks = doc_loader.create_chunks(document_splits)