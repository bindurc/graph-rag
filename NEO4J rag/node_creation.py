from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI 
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.graphs import Neo4jGraph
from langchain.schema import Document
import re
import os

loader = PyPDFLoader("data/MedicalAdvice.pdf")
pages = loader.load()
full_text = "\n".join([p.page_content for p in pages])

def clean_text(text):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  

full_text = clean_text(full_text)


splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
chunks = splitter.split_text(full_text)
documents = [Document(page_content=chunk) for chunk in chunks]


neo4j_url = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("88-DfQHXoTiTQyggc1srFdztpFW2hpvUv_XAGHJZVHo")

graph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password
)


graph.query("MATCH (n) DETACH DELETE n")



llm = ChatOpenAI(temperature=0, model="gpt-4o")
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Disease", "Symptom", "Treatment", "Diagnosis", "Prevention", "SeekHelp"],
    allowed_relationships=[
        "HAS_SYMPTOM", "HAS_DIAGNOSIS", "HAS_TREATMENT", "HAS_PREVENTION", "SEEK_HELP_WHEN"
    ],
    node_properties=False,
    relationship_properties=False
)


for doc in documents:
    graph_docs = transformer.convert_to_graph_documents([doc])
    graph.add_graph_documents(graph_docs)

print("Nodes Created")


