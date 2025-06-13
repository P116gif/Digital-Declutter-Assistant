import shutil 
import os 
import json 
import sqlite3 
from datetime import datetime
from typing import Dict, Any, List, Optional, Annotated 
from uuid import uuid4

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool 
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import (

    PyPdfLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages

try:
    import filetype
except ImportError:
    print("Installing filetype library")
    os.system("uv pip install filetype")
    import filetype 

class DocumentClassification(BaseModel):
    """Structure for document classification output"""
    classification: str = Field(description="Primary category of the document (e.g., Financial, Legal, Medical, Technical)")
    subcategory: str = Field(description="More specific classification within the primary category")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    suggested_location: str = Field(description="Sugested folder path for organising this document")
    identified_entities: List[str] = Field(description="Key entities, dates, amounts of important information extracted")
    summary: str = Field(description="Brief summary of the document content")
    keywords: List[str] = Field(description="Important keywords for search indexing")


def init_search_databse():
    """Initialise SQLite databse for document search index"""
    conn = sqlite3.connect('document_search_index.db')
    cursor = conn.cursor()

    #Create a normal table for metadata storage
    cursor.execute('''

        CREATE TABLE IF NOT EXISTS documents(

            doc_id INTEGER PRIMARY KEY,
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
            created_at TIMESTAMP,
            content_preview TEXT       
        )
    ''')

    #Create Full Text Search 5 (FTS5) table for faster searching when finding matches
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content_preview,
            keywords,
            entities,
            summary,
            classification,
            subcategory,
            content='documents',
            content_rowid='doc_id'
        )
    ''')

    #Create triggers to keep FTS5 table synchronized
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES (new.doc_id, new.content_preview, new.keywords, new.entities, new.summary, new.classification, new.subcategory);
        END;
    ''')

    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES('delete', old.doc_id, old.content_preview, old.keywords, old.entities, old.summary, old.classification, old.subcategory);
        END;
    ''')

    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES('delete', old.doc_id, old.content_preview, old.keywords, old.entities, old.summary, old.classification, old.subcategory);
            INSERT INTO documents_fts(rowid, content_preview, keywords, entities, summary, classification, subcategory)
            VALUES (new.doc_id, new.content_preview, new.keywords, new.entities, new.summary, new.classification, new.subcategory);
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
def extract_document_content(file_path: str) -> str:
    """
        Extract text content from various document types.

        Arguments:
            file_path: Path to the document to extract content from

        Returns:
            content: Extracted text content
    """

    try:
        file_type = detect_file_type(file_path)

        if file_type == 'pdf':
            loader = PyPdfLoader(file_path)
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
def classify_document_content(file_path: str, content: str) -> Dict[str, Any]:
    """
        Classify document context and extract keyword information using the llm

        Arguments:
            file_path: Path to the original file
            content: The first 5,000 characters in the file

        Returns:
            classification_result: Dictionary with classification details

    """

    try:

        llm = ChatOllama(
            model = "qwen3",
            temperatute = 0,
            base_url = "http://localhost:11434",
        )

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
            5. Extracting key entities, dates, amounts, names, or important information
            6. Creating a brief summary
            7. Identifying important keywords for search

            Focus on accuracy and provide practical organizational suggestions.
        """)

        result = structured_llm.invoke(
            classification_prompt.invoke({"content": content[:5000]})
        )

        classification_dict = {
            
            "file_path": file_path,
            "classification": result.classification,
            "subcategory": result.subcategory,
            "confidence_score": result.confidence_score,
            "suggested_location": result.suggested_location,
            "identitfied entities": result.identified_entities,
            "summary": result.summary,
            "keywords": result.keywords
        }

        return classification_dict
    
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}
    

@tool 
def create_search_index_entry(classification_result: Dict[str, Any], content: str) -> str:
    """
        Create a search index entry for the proccessed document.

        Arguments:
            classification_result: Classification documents from classify_document_content
            content: original document content
        
        Returns:
            status: success or error message
    """

    try:

        doc_id = str(uuid4())
        
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
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(texts)
            }
            chunk_metadatas.append(chunk_meta)

        vectorStore_dir = "document_vectorstore"

        if not os.path.exists(vectorStore_dir):

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
            conn = sqlite3.connect("document_search_index.db")
            cursor = conn.cursor()

            cursor.execute(''' 

                INSERT OR REPLACE INTO documents
                (doc_id, file_path, file_name, file_type, classification, subcategory, 
                confidence_score, suggested_location, summary, keywords, entities, 
                created_at, content_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',(

                doc_id,
                classification_result.get("file_path", ""),
                os.path.basename(classification_result.get("file_path","")),
                detect_file_type(classification_result.get("file_path", "")),
                classification_result.get("classification", ""),
                classification_result.get("subcategory", ""),
                classification_result.get('confidence_score', 0.0),
                classification_result.get('suggested_location', ''),
                classification_result.get('summary', ''),
                ', '.join(classification_result.get('keywords', [])),
                ', '.join(classification_result.get('identified_entities', [])),
                datetime.now().isoformat(),
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

    stop_words = set(stopwords.words("english"))
    words= word_tokenize(query)
    
    keywords = [w for w in words if w.lower() not in stop_words and w.isalnum()]
    synonms = set(keywords)
    max_synonyms_per_word = 3

    for kw in keywords:
        count = 0
        for syn in wordnet.synsets(kw):
            for lemma in syn.lemmas():
                if lemma.name() != kw:
                    synonms.add(lemma.name)
                    count += 1
                    if count >= max_synonyms_per_word:
                        break
            if count >= max_synonyms_per_word:
                break
    
    return list(synonms)


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
            return ["No documents indexed yet. Please process some documents first"]
        
        vectorStore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
        
        faiss_results = vectorStore.similarity_search_with_score(query, k=max_results)
        faiss_docs = [{

            "file_path": doc.metadata["file_path"],
            "file_id": doc.metadata["doc_id"],
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

        return sqlite_results
     
    except Exception as e:
        return [{"error": f"Error occured in fallback sql search {str(e)}"}]
    
    finally:
        conn.close()


def _sqlite_fts5_search(query: str, search_type: str ="basic", max_results: int = 5) -> List[Dict[Any, Any]]:
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

        if search_type == "phrase":
            fts_query = f'"{query}"'
        elif search_type == "boolean":
            fts_query = query
        elif search_type == "prefix":
            #prefix matching
            keywords = _extract_keywords_and_synonyms(query)
            fts_query = ' OR '.join([f'{keyword}*' for keyword in keywords])
        else:
            #basic
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
                JOIN documents_fts fts ON d.doc_id = fts.rowid
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

        Args:
            query: Search Query
            max_results: Maximum numbe of results to return

        Returns:
            results: List of matching documents with combined relevance score
    """

    conn = None
    try:

        faiss_docs = _faiss_search(query, max_results)

        sqlite_fts5_docs = 
       

        all_docs = {}





class DocumentAnalysisState(TypedDict):



def document_analyser_agent(state: DocumentAnalysisState):
    """Main document analysis agent"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def analyse_document(file_path: str) -> Dict[str, Any]:



def stream_graph_updates(user_input: str):



if __name__ == "__main__":
    print("Document Understanding Agent Initialised!")
    print("Commands: ")
    print("- analyse <file_path>: Analyse a specific document")
    print("- quit: Exit the program")


    init_search_databse()

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
                print(json.dumps(result, indent=2))
            else:
                print(f"File not found: {file_path}")

        elif user_input.startswith("search "):
            query = user_input[7:].strip()
            print(f"Searching for: {query}")
            stream_graph_updates(f"Search for documents related to: {query}")
        else:
            stream_graph_updates(user_input)