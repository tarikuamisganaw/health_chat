from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import pinecone

class ChatBot():
    def __init__(self):
        load_dotenv()
        loader = TextLoader('./health.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()

        pinecone.init(
            api_key= os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        index_name = "langchain-demo"

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Use HuggingFaceEndpoint instead of HuggingFaceHub
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, temperature=0.8, top_p=0.8, top_k=50, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        from langchain_core.prompts import PromptTemplate

        template = """
        You are a health and wellness expert. Use the following context to provide concise, practical advice based on the question asked. 
        If the answer is not in the context, say "I don't know the answer." 
        Keep your response no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(),  "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )

# Instantiate the chatbot
bot = ChatBot()

# Function for generating LLM response
# Function for generating LLM response
def generate_response(input):
    response = bot.rag_chain.invoke({"question": input})
    
    # Ensure response is processed properly
    if not response or "I don't know the answer." in response:
        return "I don't know the answer to that. Please ask another question."
    
    return response