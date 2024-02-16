import os
import pickle
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, original_filename):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
            
        def get_file_extension(uploaded_file):
            file_extension =  os.path.splitext(uploaded_file)[1].lower()
            
            return file_extension
        
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap  = 100,
                length_function = len,
            )
        
        file_extension = get_file_extension(original_filename)
   

        if file_extension == ".csv":
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ',',})
            data = loader.load()

        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)  
            data = loader.load_and_split(text_splitter)

        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split(text_splitter)


   
        '''

        from langchain.agents import load_tools
        from langchain.agents import initialize_agent
        from langchain.llms import OpenAI
        import os

        os.environ["OPENAI_API_KEY"] = "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"

        openai_api_key = os.environ.get('OPENAI_API_KEY')   

        llm = OpenAI(temperature=0)

        os.environ["SERPAPI_API_KEY"] = "2ba14a9d39bf05603e5c691ae8100bea686de68c15fdc8589d1f27eb87afa6ee"
       
        tools = load_tools(["serpapi"], llm=llm)

        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)

        response = agent({"input": query})

        from langchain.docstore.document import Document

        google_search_data =  Document(page_content=response["output"], metadata={"source": "local"})
     
        from langchain.document_loaders.merge import MergedDataLoader

        data_combined = MergedDataLoader(loaders=[data, google_search_data])

        '''
        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)
        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)

    def getDocEmbeds(self, file, original_filename):
        """
        Retrieves document embeddings
        """
        if not os.path.isfile(f"{self.PATH}/{original_filename}.pkl"):
            self.storeDocEmbeds(file, original_filename)

        # Load the vectors from the pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "rb") as f:
            vectors = pickle.load(f)
        
        return vectors
