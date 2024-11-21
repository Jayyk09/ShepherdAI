import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from helper_utils import preprocess_data, create_faiss_index, augment_query_generated, generate_multiple_queries

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

pdf_paths = ["RAG_Model/data/Matthew_Henrys_Concise_Commentary_On_The_Bible.pdf", "RAG_Model/data/The-Bible,-New-Revised-Standard-Version.pdf"]
chunks = preprocess_data(pdf_paths)

retriever = create_faiss_index(chunks)

llm = OpenAI(model="text-davinci-003", temperature=0.2, max_tokens=300)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(),
    return_source_documents=True
)

# Example query
query = "What does the Bible say about forgiveness?"
result = qa_chain.run(query)

# Print the response and sources
print("Answer:", result['result'])
print("Sources:", result['source_documents'])
