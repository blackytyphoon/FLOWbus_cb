def resp(input):
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

    persist_directory = 'db'
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

    load_dotenv()
    api_key = os.getenv("API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    custom_prompt_template = """You are FlowBus, an intelligent transportation system.
    Your responses should be in the first person, as if you are the FlowBus system itself.
    Use the following context to answer questions, but maintain your identity as FlowBus throughout.
    If you don't have the information to answer a question, say so politely as FlowBus would.

    Context:
    {context}

    Human: {question}
    FlowBus: """

    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    def process_llm_response(llm_response):
        return llm_response['result']

    query = input
    llm_response = qa_chain.invoke(query)
    return process_llm_response(llm_response)
