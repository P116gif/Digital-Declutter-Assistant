import shutil 
import os 
import sqlite3 
from datetime import datetime
from typing import Dict, Any, List, Optional, Annotated 
from uuid_utils import uuid7
import filetype 
import requests

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool 
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages


load_dotenv()

llm = ChatGroq(
        model = "qwen/qwen3-32b",
        temperature=0,
        )

class DocumentClassification(BaseModel):
    """Structure for document classification output"""
    classification: str = Field(description="Primary category of the document (e.g., Financial, Legal, Medical, Technical)")
    subcategory: str = Field(description="More specific classification within the primary category")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    suggested_location: str = Field(description="Sugested folder path for organising this document")
    identified_entities: List[str] = Field(description="Key entities, dates, amounts of important information extracted")
    summary: str = Field(description="Brief summary of the document content")
    keywords: List[str] = Field(description="Important keywords for search indexing")


def init_search_database():
    """Initialise SQLite databse for document search index"""
    conn = sqlite3.connect('document_search_index.db')
    cursor = conn.cursor()

    #Create a normal table for metadata storage
    cursor.execute('''

        CREATE TABLE IF NOT EXISTS documents(

            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            file_path TEXT UNIQUE,
            file_name TEXT,
            file_type TEXT,
            classification TEXT,
            subcategory TEXT,
            confidence_score REAL,
            suggested_location TEXT, 
            summary TEXT,
            keywords TEXT,
            entities TEXT,
            created_at TEXT,
            content_preview TEXT       
        )
    ''')

    #Create Full Text Search 5 (FTS5) table for faster searching when finding matches
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            doc_id,
            content_preview,
            keywords,
            entities,
            summary,
            classification,
            subcategory,
            content='documents',
            content_rowid='id'
        )
    ''')

    #Create triggers to keep FTS5 table synchronized
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, doc_id, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES (new.id, new.doc_id, new.content_preview, new.keywords, new.entities, new.summary, new.classification, new.subcategory);
        END;
    ''')

    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, doc_id, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES('delete', old.id, old.doc_id, old.content_preview, old.keywords, old.entities, old.summary, old.classification, old.subcategory);
        END;
    ''')

    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, doc_id, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES('delete', old.id, old.doc_id, old.content_preview, old.keywords, old.entities, old.summary, old.classification, old.subcategory);
            INSERT INTO documents_fts(rowid, doc_id, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES (new.id, new.doc_id, new.content_preview, new.keywords, new.entities, new.summary, new.classification, new.subcategory);
        END;
    ''')

    conn.commit()
    conn.close()


@tool 
def detect_file_type(file_path: str) -> str:
    """
        Detect the file type based on its extensions

        Arguments:
            file_path: Path to the file to analyse

        Returns:
            file_type: The detected file type, e.g. .'pdf', 'txt', 'docx'
    """

    try:
        kind = filetype.guess(file_path)
        if kind is None:
            _, file_type = os.path.splitext(file_path)
            return file_type[1:] if file_type else "unknown"
        return kind.extension
    except Exception as e:
        return f"Error detecting file type: {str(e)}"


@tool
def extract_document_content(file_type: str, file_path: str) -> str:
    """
        NOT FOR IMAGE FILES. Extract text content from text document types, i.e. from .pdf, .docx, .doc and .txt

        Arguments:
            file_type: the type of the file 
            file_path: Path to the document to extract content from

        Returns:
            content: Extracted text content
    """

    try:

        if file_type == 'pdf':
            loader = PyPDFLoader(file_path) #type: ignore
            pages = loader.load()
            content = "\n\n".join([page.page_content for page in pages])
        elif file_type in ['docx', 'doc']:
            loader = Docx2txtLoader(file_path)
            pages = loader.load()
            content = "\n\n".join(page.page_content for page in pages)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
        return content[:10000] #limit for processing
    
    except Exception as e:
        return f"Error extracting content: {str(e)}"


@tool
def extract_and_classify_image_content(image_path: str) -> Dict[str, Any]:
    """
        This function extracts the contents of an image, i.e., files of type .png, .jpg, .jpeg and then classifies them

        Arguments:
            file_path: the path to the image

        Returns:
            classification: a dictionary with the classification output
    """
    
    prompt = ChatPromptTemplate.from_template("""
                                              
        Sending a get request to the given url - {image_url} will give you the base64 encoding of an image. 
        The name of the file is: {file_name}
        You are to analyse and provide structured JSON output of the image with:
        1. Primary category (games, medical, financial, personal, education, legal, technical, unknown)
        2. Identify a subcategory within the primary category
        3. Provide a confidence score between [0-1] for your classification
        4. Suggested location to move the file to (format: /Pictures/{{PrimaryCategory}}/{{Subcategory}}/...)
            [*ALWAYS* Ensure that the folder structure starts with 'Pictures']
        5. Identify key entities in the image 
        6. Provide a brief summary of the image
        7. Identify important keywords that will be included in search queries for the image
        
        Guidelines:
        - Use 'unknown' category only if confidence < 0.4
        - Include dates in YYYY-MM-DD format when found
    """)

#UPLOAD IMAGE

    url = "https://934c-106-51-247-158.ngrok-free.app"

    upload_url = f"{url}/upload"
    _, ext = os.path.splitext(image_path)
    

    # Determine MIME type
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Upload file to docker
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, mime_type)}
        response = requests.post(upload_url, files=files)
    
    # Handle upload errors
    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.text}")
    
    #GET THE IMAGE ID AND JWT TOKEN
    image_id = response.json()["image_id"]
    token = response.json()["token"]

    #CREATE ENDPOINT URLS FOR REQ
    image_url = f"{url}/image/{image_id}?token={token}"

    image_llm = ChatGroq(
        model = "meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature = 0.2
    )

    strucutred_llm = image_llm.with_structured_output(DocumentClassification)

    #CALL AI
    result = strucutred_llm.invoke(
        prompt.invoke(
                {
                    "file_name": file_path,
                    "image_url": image_url
                }
            ))

    classification = {

            "file_path": image_path,
            "classification": result.classification, #type: ignore
            "subcategory": result.subcategory, #type: ignore
            "confidence_score": result.confidence_score, #type: ignore
            "suggested_location": result.suggested_location, #type: ignore
            "identitfied entities": result.identified_entities, #type: ignore
            "summary": result.summary, #type: ignore
            "keywords": result.keywords #type: ignore
    }

    requests.delete(image_url)
    return classification

    

    

@tool
def classify_document_content(file_path: str, content: str) -> Dict[str, Any]:
    """
        Classify text document context and extract keyword information using the llm

        Arguments:
            file_path: Path to the original file
            content: The first 5,000 characters in the file

        Returns:
            classification_result: Dictionary with classification details

    """

    try:

        structured_llm = llm.with_structured_output(DocumentClassification)

        classification_prompt = ChatPromptTemplate.from_template("""
        
            Analyse the following document content and provide a comprehensive classification:

            Document Content:
            {content}                                             
            
            Please classify this document by:
            1. Determining the primary category (Financial, Legal, Medical, Technical, Personal, Business, etc.)
            2. Identifying a specific subcategory within that classification
            3. Providing a confidence score (0.0-1.0) for your classification
            4. Suggesting an appropriate folder structure for organization
                [ALWAYS Ensure that the folder structure starts with one of these two directories: 'Documents' OR 'Downloads']
            5. Extracting key entities, dates, amounts, names, or important information
            6. Creating a brief summary
            7. Identifying important keywords for search

            Focus on accuracy and provide practical organizational suggestions.
            Guidelines:
                - Use 'unknown' category only if confidence < 0.4
                - Include dates in YYYY-MM-DD format when found
        """)

        result = structured_llm.invoke(
            classification_prompt.invoke({"content": content[:5000]})
        )

        classification_dict = {
            
            "file_path": file_path,
            "classification": result.classification, #type: ignore
            "subcategory": result.subcategory, #type: ignore
            "confidence_score": result.confidence_score, #type: ignore
            "suggested_location": result.suggested_location, #type: ignore
            "identitfied entities": result.identified_entities, #type: ignore
            "summary": result.summary, #type: ignore
            "keywords": result.keywords #type: ignore
        }

        return classification_dict
    
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}
    

@tool 
def create_search_index_entry(file_path: str, file_type: str, classification_result: Dict[str, Any], content: str) -> str:
    """
        Create a search index entry for the proccessed document.

        Arguments:
            file_path: A string containing the source of the file
            file_type: A string containting the type of file
            classification_result: Classification of the document received from classify_document_content
            content: original document content
        
        Returns:
            status: success or error message
    """

    try:

        doc_id = uuid7()
        
        embeddings = OllamaEmbeddings(
            model = "snowflake-arctic-embed2:568m",
            base_url="http://localhost:11434"
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100
        )

        texts = text_splitter.split_text(content)

        #creating chunk metadata
        chunk_metadatas = []

        for i,_ in enumerate(texts):
            chunk_meta = {
                **classification_result,
                "doc_id": str(doc_id),
                "chunk_index": i,
                "total_chunks": len(texts)
            }
            chunk_metadatas.append(chunk_meta)

        vectorStore_dir = "document_vectorstore"

        if not os.path.exists(vectorStore_dir):
            print("Creating directory")
            #create a new vectorstore
            vectorstore = FAISS.from_texts(
                texts,
                embeddings,
                metadatas=chunk_metadatas
            )

            vectorstore.save_local(vectorStore_dir)
        else:
            #load existing and append chunks

            vectorstore = FAISS.load_local(vectorStore_dir, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_texts(
                texts,
                metadatas=chunk_metadatas
            )
            vectorstore.save_local(vectorStore_dir)

        
        try:
            conn = sqlite3.connect('document_search_index.db')
            cursor = conn.cursor()

            cursor.execute(''' 

                INSERT OR REPLACE INTO documents
                (doc_id, file_path, file_name, file_type, classification, subcategory, 
                confidence_score, suggested_location, summary, keywords, entities, 
                created_at, content_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',(

                str(doc_id),
                file_path,
                str(os.path.basename(file_path)),
                file_type,
                classification_result.get("classification", ""),
                classification_result.get("subcategory", ""),
                float(classification_result.get('confidence_score', 0.0)),
                classification_result.get('suggested_location', ''),
                classification_result.get('summary', ''),
                ', '.join(classification_result.get('keywords', [])),
                ', '.join(classification_result.get('identified_entities', [])),
                str(datetime.now().isoformat()),
                content[:500]
            ))

            conn.commit()
            return "Search Index successfully created."
        
        except Exception as e:
            conn.rollback()
            return f"Error creating search index: {str(e)}"
        finally:
            conn.close()
        
    
    except Exception as e:
        return f"Error creating search index: {str(e)}"


@tool 
def move_file_to_suggested_location(file_path: str, suggested_location: str) -> str: 
    """
        Move the file to the suggested location

        Arguments:
            file_path: the current file path
            suggested_location: the destination path

        Returns:
            new_path: New file location or an error message
    """

    try:
        # Create destination directory if it doesn't exist
        
        append = str(os.environ.get("MAIN_DIRECTORY"))

        if append not in suggested_location:
            suggested_location = append + suggested_location

        os.makedirs(suggested_location, exist_ok=True)
        
        filename = os.path.basename(file_path)
        base, ext = os.path.splitext(filename)
        count = 1
        new_path = suggested_location
        
        # Handle file name conflicts
        while os.path.exists(new_path):
            new_filename = f"{base}_{count}{ext}"
            new_path = os.path.join(suggested_location, new_filename)
            count += 1
        
        shutil.move(file_path, new_path)
        return new_path
    
    except Exception as e:
        return f"Error moving file: {str(e)}"


def _extract_keywords_and_synonyms(query: str) -> List[str]:

    
    stop_words = set(stopwords.words('english'))

    #nltk.download('stopwords')
    #nltk.download('punkt_tab')
    #nltk.download('punkt')

    words = word_tokenize(query)

    
    keywords = [w for w in words if w.lower() not in stop_words and w.isalnum()]
    
    synonyms = set(keywords)
    max_synonyms_per_word = 3

    #nltk.download('wordnet')
    for kw in keywords:
        count = 0
        for syn in wordnet.synsets(kw):
            for lemma in syn.lemmas(): #type: ignore
                lemma_name = lemma.name().lower().replace('_', ' ') 
                if lemma_name != kw.lower():
                    synonyms.add(lemma_name)
                    count += 1
                    if count >= max_synonyms_per_word:
                        break
            if count >= max_synonyms_per_word:
                break
    
    return list(synonyms)


def _faiss_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
        Helper function to complete a vector search in the faiss index and modularise the code.

        Arguments:
            query: the initial query of the user
            max_results: the number of results to be returned

        Returns:
            faiss_docs: a list of dictionary items that contain the file_path, file_id and score of a document
            OR
            error: on failure to execute the code block 
    
    """
    try:

        embeddings = OllamaEmbeddings(
            model = "snowflake-arctic-embed2:568m",
            base_url="http://localhost:11434"
        )

        vectorstore_dir = "document_vectorstore"

        if not os.path.exists(vectorstore_dir):
            return [{"error": "No documents indexed yet. Please process some documents first"}]
        
        vectorStore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
        
        faiss_results = vectorStore.similarity_search_with_score(query, k=max_results)
        faiss_docs = [{

            "doc_id": doc.metadata["doc_id"],
            "score": float(score),
            "source": "vector"

        }for doc, score in faiss_results] 

        
        return faiss_docs
    
    except Exception as e:
        return [{"error": f"Error occurred while doing a faiss search: {str(e)}"}]

def _sqlite_fallback_search(query: str, max_results: int = 5) -> List[Dict[Any, Any]]:
     
    """
        Helper function to complete a full text search in the sqlite db in the case that the fts5 function fails to execute.

        Arguments:
            query: the initial query of the user
            max_results: the number of results to be returned

        Returns:
            sqlite_results: a list of dictionary items that contain the column name : row value for each search result
            OR
            error: on failure to execute the code block 
    
    """
     
    try:
         #Split query into keyword and metadata search
        conn = sqlite3.connect("documents_search_index.db")
        cursor = conn.cursor()

        keywords = _extract_keywords_and_synonyms(query)

        #split query into terms for keyword search
        terms = [f'%{kw}%' for kw in keywords]

        query_terms = []
        params = []

        for term in terms:
            query_terms.extend([
                "content_preview LIKE ?",
                "keywords LIKE ?",
                "entities LIKE ?",
                "summary LIKE ?"
            ])
            params.extend([term] * 4)

        sql = f"""
            SELECT *,
                (CASE WHEN classification LIKE ? THEN 1 ELSE 0 END) + 
                (CASE WHEN subcategory LIKE ? THEN 1 ELSE 0 END)

            AS metadata_score
                FROM documents
                WHERE {'OR'.join(query_terms) if query_terms else '1=1'}
                ORDER BY metatdata_score DESC, confidence_SCORE DESC
                LIMIT ? 
        """

        params += [f'%{query}%', f'%{query}%', max_results]

        cursor.execute(sql, params)
        sqlite_results = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        print("Fallback\n ", sqlite_results)
        return sqlite_results
     
    except Exception as e:
        return [{"error": f"Error occured in fallback sql search {str(e)}"}]
    
    finally:
        conn.close()


def _sqlite_fts5_search(query: str, max_results: int = 5) -> List[Dict[Any, Any]]:
    """
        Helper function to complete an advanced fts function and different search types.

        Arguments:
            query: the initial query of the user
            max_results: the number of results to be returned

        Returns:
            sqlite_results: a list of dictionary items that contain the column name : row value for each search result
            OR
            error: on failure to execute the code block 
    
    """
    try:
        
        conn = sqlite3.connect("document_search_index.db")
       
        cursor = conn.cursor()
 

        keywords = _extract_keywords_and_synonyms(query)
        fts_query = ' OR '.join(keywords)

        sql = """

            SELECT
                d.*,
                fts.rank as relevance_score,
                (
                    (CASE WHEN d.classification LIKE ? THEN 2 ELSE 0 END) + 
                    (CASE WHEN d.subcategory LIKE ? THEN 1 ELSE 0 END) +
                    (fts.rank * - 1)
                
                ) AS combined_score
                FROM documents d
                JOIN documents_fts fts ON d.doc_id = fts.doc_id
                WHERE documents_fts MATCH ?
                ORDER BY combined_score DESC, d.confidence_score DESC
                LIMIT ?
            """
        
        params = [f'%{query}%', f'%{query}%', fts_query, max_results]
        
        try:
            cursor.execute(sql, params)
            sqlite_results = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
            
            return sqlite_results
        
        finally:
            conn.close()

    except Exception as e:
        return [{"error": f"Error occurred while running sqlite fts5 search: {str(e)}"}]
    
    finally:
        conn.close()
     
    
    


@tool 
def search_documents(query: str, max_results: int = 5) -> List[Dict[str, Any] | str]:
    """
        Hybrid search combining semantic similarity and SQLite metadata / keyword search

        Arguments:
            query: Search Query
            max_results: Maximum number of results to return

        Returns:
            results: List of matching documents with combined relevance score
    """

    try:

        faiss_docs = _faiss_search(query, max_results)

        sqlite_docs = _sqlite_fts5_search(query, max_results=max_results)

        if "error" in sqlite_docs:
            sqlite_docs = _sqlite_fallback_search(query, max_results)


        all_docs = {}

        #process faiss results
        for idx, doc in enumerate(faiss_docs):
            key = doc.get('doc_id')
            
            all_docs[key] = {
                **doc,
                "vector_rank": idx+1,
                "sql_rank": None,
                "file_path": None,
                "combined_score": 0
            }

        #process sqlite results
        for idx, doc in enumerate(sqlite_docs):
            key = doc["doc_id"]

            if key in all_docs:
                all_docs[key]["sql_rank"] = idx + 1
                all_docs[key]["file_path"] = doc["suggested_location"]
            else:
                all_docs[key] = {
                    "doc_id": key,
                    "file_path": doc["suggested_location"],
                    "score": doc["confidence_score"],
                    "source": "sql",
                    "vector_rank": None,
                    "sql_rank": idx + 1,
                    "combined_score": 0
                }

        #Calculating Reciprocal Rank Fusion scores
        rrf_k = 60
        
        for doc in all_docs.values():
            
            vector_rrf = 1 / (rrf_k + doc["vector_rank"]) if doc["vector_rank"] else 0

            sql_rrf = 1 / (rrf_k + doc["sql_rank"]) if doc["sql_rank"] else 0

            doc["combined_score"] = vector_rrf + sql_rrf

        sorted_results = sorted(all_docs.values(),
                                key=lambda x: x["combined_score"],
                                reverse=True)[:max_results]

        results = []

        for doc in sorted_results:
            results.append({
                "doc_id": doc.get("doc_id"),
                "file_path": doc.get("file_path"),
                "combined_score": doc.get("combined_score")
            })

        return results 
    
    except Exception as e:
        return [{"error": f"There was an error running search_documents function: {str(e)}"}]



class DocumentAnalysisState(TypedDict):
    """State for langgraph agent """
    messages: Annotated[list, add_messages]
    file_path: Optional[str]
    content: Optional[str]
    classfication_result: Optional[Dict[str, Any]]
    analysis_complete: bool

# Initialize tools
tools = [
    detect_file_type,
    extract_document_content,
    classify_document_content,
    create_search_index_entry,
    move_file_to_suggested_location,
    search_documents,
    extract_and_classify_image_content
]

llm_with_tools = llm.bind_tools(tools)

# Create tool node
tool_node = ToolNode(tools=tools)

def document_analyser_agent(state: DocumentAnalysisState):
    """Main document analysis agent"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(DocumentAnalysisState)

graph_builder.add_node("agent", document_analyser_agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools":"tools", END:END}
)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()

def analyse_document(file_path: str) -> Dict[str, Any]:
    """
    Complete the entire document analysis workflow

    Arguments:
        file_path: the source location of the file to be analysed

    Returns:
        analysis_result: complete analysis including classification and indexing
    """

    user_message = f""" 
    
    Please analyze the document at: {file_path}
    
    Follow these steps:
    1. Find the file type of the document
    2. Extract the document content  
    3. Classify the document content and extract key information
    4. ALWAYS create a search index entry for the document
    5. Provide the classification result in the specified JSON format
    6. FINALLY, Move the file to the suggested location using the move_file tool
    
    The final output should be in this JSON format:
    {{
        "classification": "Primary category",
        "subcategory": "Specific subcategory", 
        "confidence_score": 0.95,
        "suggested_location": "/Documents/Category/Subcategory/",
        "identified_entities": ["entity1", "entity2", "entity3"]
    }}    
    """

    try:

        #initialise the db
        init_search_database()

        result = graph.invoke({
            "messages": [{"role": "user", "content": user_message}],
            "file_path": file_path,
            "content": None,
            "classification_result": None,
            "analysis_complete": False
        })

        return {"status": "success", "messages": result["messages"]}
    except Exception as e:
        return {"error": f"Error occured when running analyse document {str(e)}"}
    


def stream_graph_updates(user_input: str):
    """Stream AI output to terminal"""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if "messages" in value and value["messages"]:
                print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    print("Document Understanding Agent Initialised!")
    print("Commands: ")
    print("- analyse <file_path>: Analyse a specific document")
    print("- search <query>: To search for something in the database + vectorstore")
    print("- quit / exit / q: Exit the program")


    init_search_database()

    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input.startswith("analyse ") or user_input.startswith("analyze "):
            file_path = user_input[8:].strip()
            if os.path.exists(file_path):
                print(f"Analysing Document: {file_path}")
                result = analyse_document(file_path)
            else:
                print(f"File not found: {file_path}")
        elif user_input.startswith("search "):
            query = user_input[7:].strip()
            print(f"Searching for: {query}")
            stream_graph_updates(f"Search for documents related to: {query}")
        else:
            stream_graph_updates(user_input)