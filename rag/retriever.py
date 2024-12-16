import uuid
import pickle

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class Retriever:
    def __init__(self, embeddings, chroma_persist_directory="./chroma_db", store_pickle=None):
        self.embeddings = embeddings
        
        if store_pickle:
            try:
                with open(store_pickle, "r+b") as f:
                    self.store = pickle.load(f)
            except FileNotFoundError:
                self.store = InMemoryStore()
        else:
            self.store = InMemoryStore()

        self.id_key = "doc_id"

        self.vectorstore = Chroma(
            collection_name="mm_rag",
            embedding_function=embeddings,
            persist_directory=chroma_persist_directory
        )

        self.store_pickle = store_pickle
        self.chroma_persist_directory = chroma_persist_directory

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )

    def save(self):
        with open(self.store_pickle, "wb") as f:
            pickle.dump(self.store, f)

    def add_documents(self, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={self.id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        self.vectorstore.add_documents(summary_docs)
        self.store.mset(list(zip(doc_ids, doc_contents)))

    def __call__(self, query, limit=5):
        return self.retriever.invoke(query)

   
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("./data/descriptions_dataset.csv")
    df = df.sample(40000)
    df = df.dropna()

    text_summaries = df["TITLE"].tolist()
    texts = df.drop(columns=["TITLE", "PRODUCT_ID", "PRODUCT_LENGTH", "PRODUCT_TYPE_ID"]).apply(lambda row: row.to_json(), axis=1).tolist()

    retriever = Retriever(
        HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'), 
        chroma_persist_directory="./chroma_desc_db",
        store_pickle="chroma_db/retriever_desc_store.pkl"
    )
    retriever.add_documents(text_summaries, texts)

    print(retriever("water bottle", limit=3))
    retriever.save()