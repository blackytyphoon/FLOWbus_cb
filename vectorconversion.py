def vector():
    import os
    import warnings
    from dotenv import load_dotenv
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    import google.generativeai as genai
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    from langchain.prompts import PromptTemplate

    # Suppress LangChain deprecation warnings
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    
    load_dotenv()
    api_key = os.getenv("API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
    # Initialize the LLM with Gemini
    loader = TextLoader(r"C:\Users\ssjis\fbcb\data\data.txt", encoding='utf-8')
    documents = loader.load()
       
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    persist_directory = 'db'
    
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

vector()
print("Done")