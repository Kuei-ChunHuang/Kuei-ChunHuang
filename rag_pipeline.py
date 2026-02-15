from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class RAGPipeline:
    def __init__(self, collection):
        embeddings = OpenAIEmbeddings()
        self.vector_store = MongoDBAtlasVectorSearch(collection, embeddings)
        self.llm = OpenAI(temperature=0)

    def query(self, user_input):
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever())
        return qa.run(user_input)

if __name__ == "__main__":
    print("RAG Pipeline with MongoDB Atlas ready.")
