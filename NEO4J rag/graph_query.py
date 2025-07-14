import os
from langchain.prompts import PromptTemplate
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector


neo4j_url = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")


llm = ChatOpenAI(temperature=0, model="gpt-4o")

graph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password,
    database="neo4j",
    node_label="Disease",  
    text_node_properties=["id", "text"],
    embedding_node_property="embedding",
    index_name="vector_index_v2",
    keyword_index_name="entity_index_v2",
    search_type="hybrid"
)


schema = graph.get_schema

template = """
Task: You are a knowledgeable assistant specialized in analyzing Neo4j graph database schemas.

Your task is to provide accurate and concise natural language answers to questions by strictly referencing the provided schema.

Instructions:
Use only relationship types and properties provided in schema.
Do not use other relationship types or properties that are not provided.
Use only the node labels, relationship types, and properties explicitly mentioned in the schema.
Do not fabricate, assume, or infer any labels, relationships, or properties not present in the schema.
Your answer should clearly explain the relationships between entities based on the schema structure.
Keep your answer factual, concise, and easy to understand.

schema:
{schema}

Note: Do not include explanations or apologies in your answers.
Provide a structured, well-explained answer that reflects the graph's logic.

Question: {question}
"""

question_prompt = PromptTemplate(
    template=template,
    input_variables=["schema", "question"]
)

qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=question_prompt,
    verbose=True,
    allow_dangerous_requests=True
)

question = "list all the disease and explain the response to a layman"
response = qa_chain.invoke({"query": question})

print("\nQuery Result:\n", response["result"])
