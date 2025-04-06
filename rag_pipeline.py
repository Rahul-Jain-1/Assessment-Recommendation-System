from langchain_chroma import Chroma
from prepare import *
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

class RAG_pipeline():
    def __init__(self, path = "./VectorDB"):

        """Using PersisentClient to save the DB in local so that we """

        self.path = path 
        self.embedding_model = "models/text-embedding-004"
        self.embedding_fuction = self.get_embeddings_function()
        self.collection = self.configure_db_collection()
    
    def configure_db_collection(self):

        """Creating Collection if the collection doesn't exists, otherwise retrieving the existing Collection"""
        try:
            if os.path.exists(self.path) and os.listdir(self.path):
                print("Loading existing vector store from disk...")
                collection = Chroma(persist_directory=self.path, embedding_function=self.embedding_fuction)
                return collection
        except Exception:
            return None

        
    def get_embeddings_function(self):

        """Function to generate the llama3 embeddings of a single text"""

        google_ef = GoogleGenerativeAIEmbeddings(model=self.embedding_model)
        return google_ef
    
    def build_document(self, df):
        separator_id = "1f1f7b1e-6a6b-48d4-9f4e-0b9db5b3d72c"
        with open("documents.txt", "w", encoding="utf-8") as f:
            # Convert the Series to a list of strings and join them with "\n\n"
            f.write(f"\n{separator_id}\n".join(df["Embedding text"].astype(str).tolist()))
        

        raw_documents = TextLoader("documents.txt", encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator=f"\n{separator_id}\n")
        documents = text_splitter.split_documents(raw_documents)
        
        for idx,doc in enumerate(documents):
                doc.metadata = {
                    "id": idx,
                    "Assessment Name": df['Assessment Name'][idx],
                    "URL": df['URL'][idx],
                    "Remote Testing": df["Remote Testing"][idx],
                    "Adaptive/IRT": df["Adaptive/IRT"][idx],
                    "Test Type": df["Test Type"][idx],
                    "Duration": df['Duration'][idx]
                 }
        return documents
            
    def store_data_in_db(self, documents):

        """Going Iteratively to generate embeddings of all the Description in the provided df"""
        try:
            print("Creating new vector store...")
            
            self.collection = Chroma.from_documents(documents, embedding=self.embedding_fuction, persist_directory=self.path)
                 
            print("All the data are successfully stored in the DB")
        except Exception as e:
            print("Exception !!!")
            print(e)     
    
    def process(self ):

        df = pd.read_csv("shl_df.csv")
        self.configure_db_collection()
        collection = self.collection
        if collection is None:
            documents = self.build_document(df)
            self.store_data_in_db(documents)
            collection=self.collection

       
    
    def query_rag(self, query, top_k=10):
        print(f"value of colection is {self.collection}")
        result=self.collection.similarity_search_with_score(query, k = top_k)
        print(f"result of rag is {result}\n\n")
        final_metatdata = []
        for doc,score in result:
            if score >= 0.66:
                continue
            
            doc.metadata["score"]=score
            print(f"score is {score}")
            metadata = doc.metadata
            final_metatdata.append(
                {
                    "assessment name": metadata.get("Assessment Name"),
                    "url": metadata.get("URL"),
                    "Remote Testing Support": metadata.get("Remote Testing"),
                    "Adaptive/IRT Support": metadata.get("Adaptive/IRT"),
                    "duration": metadata.get("Duration"),
                    "test type": metadata.get("Test Type"),
                }
            )
        
        print(f"final_result is \n {final_metatdata}")    
        return final_metatdata

# if __name__ == "__main__":
#     rag = RAG_pipeline()
#     rag.process()