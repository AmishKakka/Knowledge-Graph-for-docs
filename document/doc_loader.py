from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint
from time import time
import uuid
from collections import defaultdict
from pathlib import Path


class DocumentLoader:
    def __init__(self):
        '''
            Initializes the DocumentLoader with RecursiveCharacterTextSplitter.
        '''
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                            chunk_overlap=250,
                                                            length_function=len,
                                                            separators=["\n\n", "\n"])

    def get_doc_id(self):
        '''Returns a unique document ID.'''
        return uuid.uuid4()

    def lazy_load_and_split(self, doc_paths: List[str]):
        '''
            Lazily loads and splits one or more documents into chunks.

            Accepts a single path or a list of paths and returns a flat list
            of document chunks.
        '''
        if not isinstance(doc_paths, (list, tuple)):
            doc_paths = [doc_paths]

        print("Lazy loading and splitting documents...")
        loaded_docs = []
        for path in doc_paths:
            loader = PyMuPDFLoader(path)
            self.loader = loader
            self.doc_id = self.get_doc_id()
            for doc in loader.lazy_load():
                loaded_docs.extend(self.text_splitter.split_documents([doc]))
        return loaded_docs

    def create_chunks(self, chunks: list):
        '''
            Creates chunks with metadata from the list of document splits.
            
            **Returns a list of chunks with metadata and a list of texts**.
        '''
        chunks_with_metadata = []
        list_of_texts = []
        counters = defaultdict(int)

        for c in chunks:
            file_path = c.metadata.get('file_path', 'unknown')
            idx = counters[file_path]
            counters[file_path] += 1
            ns = Path(file_path).stem
            chunk_id = f"{ns}:{idx}"

            chunks_with_metadata.append({
                'text': c.page_content,
                'file_path': file_path,
                'page': c.metadata.get('page'),
                'chunkId': chunk_id,
                'chunkIndex': idx
            })
            list_of_texts.append(c.page_content)
        print(f"Created {len(chunks)} chunks with metadata...")
        return chunks_with_metadata, list_of_texts

  
# if __name__ == "__main__":
#     doc_loader = DocumentLoader()

#     # st = time()
#     # document_splits = doc_loader.load_and_split()
#     # et = time()
#     # print("Total time taken: ", et-st)

#     st = time()
#     document_splits = doc_loader.lazy_load_and_split("AttentionPaper.pdf")
#     et = time()
#     print("Total time taken to load, split, chunk document : ", et-st)
#     print("\nTotal document splits: ", len(document_splits))
  
#     chunks = doc_loader.create_chunks(document_splits)