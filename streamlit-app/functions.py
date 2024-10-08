from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import os
import tempfile
import uuid
import pandas as pd
import re

def clean_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

def get_pdf_text(upoaded_file):
    try:
        input_file = upoaded_file.read()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()
        
        loader = PyPDFLoader(temp_file.name)
        
        documents = loader.load_and_split()
        
        return documents
    except Exception as e:
        print(e)
    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(temp_file.name)

def split_document(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                            chunk_overlap=200,
                                            length_function=len,
                                            separators=["\n\n", "\n", " "])
    chunks = text_splitter.split_documents(documents)
    
    return chunks

def get_embedding(api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
    return embeddings

def create_vectorstore(chunks, embedding_function, file_name, vectorstore_path="db"):
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    unique_ids = set()
    unique_chunks = []
    
    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)
    
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        ids=list(unique_ids),
                                        embedding=embedding_function, 
                                        persist_directory=vectorstore_path)
    vectorstore.persist()
    return vectorstore

def create_vectorstore_from_texts(documents, api_key, file_name):
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3 define embedding function
    embedding_function = get_embedding(api_key)

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    return vectorstore

def load_vectorstore(file_name, api_key, vectorstore_path="db"):
    embedding_function = get_embedding(api_key)
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the research article"""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.
def query_document(vectorstore, query, api_key):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    retriever=vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
        )

    structured_response = rag_chain.invoke(query)
    df = pd.DataFrame([structured_response.dict()])

    # Transforming into a table with two rows: 'answer' and 'source'
    answer_row = []
    source_row = []
    reasoning_row = []

    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['sources'])
        reasoning_row.append(df[col][0]['reasoning'])

    # Create new dataframe with two rows: 'answer' and 'source'
    structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
    return structured_response_df.T