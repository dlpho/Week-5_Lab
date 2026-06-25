from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from core.llmops import OpsCallbackHandler
# Import your Week 4 guardrail functions (input_guard, output_guard) here

# Initialize shared components
llm = Ollama(model="gemma3:1b", callbacks=[OpsCallbackHandler()])
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, return_messages=False)

PROMPT = ChatPromptTemplate.from_template("""
Answer using the context and history.
Context: {context}
History: {history}
Question: {question}
""")

def process_pdf(file_path: str):
    """Processes uploaded PDF and updates ChromaDB."""
    # Insert your Week 3 PyPDFLoader and SemanticChunker logic here
    pass

def generate_chat_stream(query: str):
    """Yields chunks of the response for SSE and Streamlit streaming."""
    try:
        # 1. Input Guardrails
        clean_query = input_guard(query)
        
        # 2. RAG Retrieval
        results = db.similarity_search(clean_query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        history = memory.load_memory_variables({})["history"]
        
        # 3. Stream LLM Response
        chain = PROMPT | llm
        full_response = ""
        
        for chunk in chain.stream({"context": context, "history": history, "question": clean_query}):
            # 4. Output Guardrail (applied per chunk or accumulated)
            safe_chunk = output_guard(chunk) 
            full_response += safe_chunk
            yield safe_chunk
            
        # 5. Save to memory after stream completes
        memory.save_context({"input": clean_query}, {"output": full_response})

    except ValueError as e:
        # Yield guardrail block messages
        yield str(e)