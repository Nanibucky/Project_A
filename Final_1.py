"""
Hybrid Agent System
A sophisticated system that combines SQL database querying with document retrieval capabilities.
"""

import os
import requests
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.core.schema import Node, NodeWithScore
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.llms.openai import OpenAI

# Document processing
import PyPDF2
import docx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hybrid_agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

###################
# Configuration
###################

@dataclass
class Config:
    """Configuration settings for the Hybrid Agent system."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.0
    CHUNK_SIZE: int = 500  # Smaller chunks for better precision
    CHUNK_OVERLAP: int = 100  # Overlap for context
    CACHE_SIZE: int = 100
    SQL_CONFIDENCE_THRESHOLD: float = 0.6
    RAG_CONFIDENCE_THRESHOLD: float = 0.4
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour

###################
# Data Models
###################

@dataclass
class QueryResult:
    """Represents the result of a query."""
    content: str
    confidence: float
    source: str  # 'sql' or 'rag'
    metadata: Dict
    timestamp: datetime = datetime.now()

@dataclass
class SchemaInfo:
    """Database schema information."""
    tables: Dict[str, List[str]]  # table_name -> [column_names]
    relationships: Dict[str, List[str]]  # table_name -> [related_tables]
    
    def has_table(self, table_name: str) -> bool:
        return table_name.lower() in {t.lower() for t in self.tables.keys()}
    
    def has_column(self, column_name: str) -> bool:
        return any(
            column_name.lower() in {c.lower() for c in columns}
            for columns in self.tables.values()
        )

###################
# Document Processing
###################

def load_documents(docs_path: Path) -> List[Document]:
    """Load documents from the specified directory."""
    documents = []
    
    for file_path in docs_path.glob("*.*"):
        try:
            if file_path.suffix.lower() == '.pdf':
                doc = load_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                doc = load_docx(file_path)
            else:
                continue
                
            if doc:
                # Split documents into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.CHUNK_OVERLAP
                )
                split_docs = text_splitter.split_documents([doc])
                documents.extend(split_docs)
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
    
    return documents

def load_pdf(file_path: Path) -> Optional[Document]:
    """Load a PDF document."""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            content = ' '.join(
                page.extract_text()
                for page in reader.pages
            )
            return Document(
                page_content=content,
                metadata={'source': file_path.name, 'type': 'pdf'}
            )
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return None

def load_docx(file_path: Path) -> Optional[Document]:
    """Load a DOCX document."""
    try:
        doc = docx.Document(file_path)
        content = ' '.join(
            paragraph.text
            for paragraph in doc.paragraphs
        )
        return Document(
            page_content=content,
            metadata={'source': file_path.name, 'type': 'docx'}
        )
    except Exception as e:
        logger.error(f"Error loading DOCX {file_path}: {e}")
        return None

###################
# Retrievers
###################

class SQLRetriever:
    """SQL database retriever."""
    def __init__(self, db: SQLDatabase, config: Config):
        self.db = db
        self.config = config
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )
        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="openai-tools",
            verbose=True
        )
    
    def retrieve(self, query: str) -> QueryResult:
        try:
            result = self.agent.run(query)
            return QueryResult(
                content=result,
                confidence=1.0,
                source='sql',
                metadata={'query_type': 'sql'}
            )
        except Exception as e:
            logger.error(f"SQL retrieval error: {e}")
            return QueryResult(
                content=str(e),
                confidence=0.0,
                source='sql',
                metadata={'error': str(e)}
            )

class RAGRetriever:
    """RAG (Retrieval-Augmented Generation) retriever."""
    def __init__(self, documents: List[Document], config: Config):
        self.config = config
        self.documents = documents
        self.dense_retriever = self._init_dense_retriever()
        self.sparse_retriever = self._init_sparse_retriever()
        self.ensemble_retriever = self._init_ensemble_retriever()
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )
    
    @lru_cache(maxsize=Config.CACHE_SIZE)
    def _init_dense_retriever(self):
        embeddings = OpenAIEmbeddings(api_key=self.config.OPENAI_API_KEY)
        return FAISS.from_documents(
            self.documents,
            embeddings
        ).as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 chunks
    
    def _init_sparse_retriever(self):
        return BM25Retriever.from_documents(self.documents, k=5)  # Retrieve top 5 chunks
    
    def _init_ensemble_retriever(self):
        return EnsembleRetriever(
            retrievers=[self.sparse_retriever, self.dense_retriever],
            weights=[self.config.SPARSE_WEIGHT, self.config.DENSE_WEIGHT]
        )
    
    def retrieve(self, query: str) -> QueryResult:
        try:
            # Retrieve documents using the ensemble retriever
            docs = self.ensemble_retriever.invoke(query)
            if not docs:
                return QueryResult(
                    content="No relevant information found.",
                    confidence=0.0,
                    source='rag',
                    metadata={'found_docs': 0}
                )
            
            # Filter irrelevant chunks based on query
            filtered_docs = self._filter_docs(query, docs)
            if not filtered_docs:
                return QueryResult(
                    content="No relevant information found.",
                    confidence=0.0,
                    source='rag',
                    metadata={'found_docs': 0}
                )
            
            # Summarize the filtered documents
            summary = self._summarize_docs(query, filtered_docs)
            return QueryResult(
                content=summary,
                confidence=0.8,
                source='rag',
                metadata={'found_docs': len(filtered_docs)}
            )
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return QueryResult(
                content=str(e),
                confidence=0.0,
                source='rag',
                metadata={'error': str(e)}
            )
    
    def _filter_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Filter out irrelevant documents based on the query."""
        filtered_docs = []
        for doc in docs:
            # Check if the query terms are present in the document
            if any(term.lower() in doc.page_content.lower() for term in query.split()):
                filtered_docs.append(doc)
        return filtered_docs
    
    def _summarize_docs(self, query: str, docs: List[Document]) -> str:
        """Summarize the filtered documents to extract relevant information."""
        # Combine the top 3 chunks
        combined_content = "\n".join(doc.page_content for doc in docs[:3])
        
        # Use a focused summarization prompt
        prompt = f"""
        You are a helpful assistant. Summarize the following text to answer the query: "{query}".
        Text: {combined_content}
        Summary:
        """
        
        # Generate the summary using the LLM
        response = self.llm.invoke(prompt)
        return response.content

###################
# Query Router
###################

class QueryRouter:
    """Routes queries to appropriate retrievers based on schema and query patterns."""
    def __init__(self, schema_info: SchemaInfo, config: Config):
        self.schema_info = schema_info
        self.config = config
        self.sql_patterns = {
            'aggregation': r'\b(count|sum|average|avg|total|mean)\b',
            'comparison': r'\b(more than|less than|greater|highest|lowest)\b',
            'temporal': r'\b(when|date|year|month|time|period)\b',
            'numerical': r'\b(how many|how much|price|cost|revenue|sales)\b',
            'music': r'\b(artist|track|album|song|genre|playlist)\b',
            'customer': r'\b(customer|client|buyer|purchase|spending)\b',
            'genre': r'\b(genre|category|type|style)\b'
        }
        self.rag_patterns = {
            'conceptual': r'\b(what is|explain|describe|how does)\b',
            'research': r'\b(paper|research|study|analysis|findings)\b',
            'procedural': r'\b(how to|steps|process|procedure)\b'
        }

    def route_query(self, query: str) -> Tuple[str, float]:
        """Determine the best route for the query using schema and patterns."""
        query = query.lower()

        # Check schema match first
        schema_score = self._check_schema_match(query)
        if schema_score > self.config.SQL_CONFIDENCE_THRESHOLD:
            logger.info(f"Query matches schema with score {schema_score}. Routing to SQL.")
            return 'sql', schema_score

        # Pattern matching
        sql_score = self._check_patterns(query, self.sql_patterns)
        rag_score = self._check_patterns(query, self.rag_patterns)

        # Handle case where both scores are zero
        if sql_score == 0 and rag_score == 0:
            logger.info("No patterns matched. Defaulting to RAG.")
            return 'rag', 0.0  # Default to RAG if no patterns match

        # Normalize scores
        total_score = sql_score + rag_score
        if total_score == 0:
            logger.info("No patterns matched. Defaulting to RAG.")
            return 'rag', 0.0  # Default to RAG if no patterns match

        if sql_score > rag_score:
            return 'sql', sql_score / total_score
        return 'rag', rag_score / total_score

    def _check_schema_match(self, query: str) -> float:
        """Check if the query matches the database schema."""
        score = 0.0
        for table in self.schema_info.tables:
            if table.lower() in query:
                score += 0.5
            for column in self.schema_info.tables[table]:
                if column.lower() in query:
                    score += 0.3
        return min(score, 1.0)

    def _check_patterns(self, query: str, patterns: dict) -> float:
        """Check if the query matches any of the given patterns."""
        score = 0.0
        for pattern in patterns.values():
            if re.search(pattern, query):
                score += 1.0
        return score

###################
# Query Decomposer
###################

class QueryDecomposer:
    """Decomposes a complex query into sub-queries."""
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )
    
    def is_complex_query(self, query: str) -> bool:
        """Determine if the query is complex and needs decomposition."""
        # A query is complex if it contains multiple clauses or asks for multiple pieces of information
        return len(query.split(",")) > 1 or len(query.split(" and ")) > 1
    
    def decompose(self, query: str) -> List[str]:
        """Break down a complex query into individual sub-queries."""
        prompt = f"""
        You are a helpful assistant. Break down the following complex query into clear and actionable sub-queries:
        Query: "{query}"
        Sub-queries:
        """
        response = self.llm.invoke(prompt)
        sub_queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        return sub_queries

###################
# Main Agent
###################

class HybridAgent:
    """Main Hybrid Agent class combining SQL and RAG capabilities."""
    
    def __init__(
        self,
        database_uri: str,
        docs_directory: str,
        config: Optional[Config] = None
    ):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db = SQLDatabase.from_uri(database_uri)
        self.schema_info = self._extract_schema_info()
        self.documents = load_documents(Path(docs_directory))
        
        # Initialize retrievers, router, and decomposer
        self.sql_retriever = SQLRetriever(self.db, self.config)
        self.rag_retriever = RAGRetriever(self.documents, self.config)
        self.router = QueryRouter(self.schema_info, self.config)
        self.decomposer = QueryDecomposer(self.config)
    
    def _extract_schema_info(self) -> SchemaInfo:
        """Extract database schema information."""
        try:
            # Get list of valid tables
            tables = {}
            relationships = {}
            
            # Get all usable table names
            table_names = self.db.get_usable_table_names()
            logger.info(f"Found tables: {table_names}")
            
            # Process each table
            for table_name in table_names:
                try:
                    # Get table info
                    table_info = self.db.get_table_info(table_name)
                    columns = [col['name'] for col in table_info]
                    tables[table_name] = columns
                    
                    # Initialize relationships (can be enhanced later)
                    relationships[table_name] = []
                    
                    logger.info(f"Processed table {table_name} with columns: {columns}")
                except Exception as e:
                    logger.error(f"Error processing table {table_name}: {e}")
                    continue
            
            return SchemaInfo(tables=tables, relationships=relationships)
            
        except Exception as e:
            logger.error(f"Error extracting schema info: {e}")
            # Return empty schema info if extraction fails
            return SchemaInfo(tables={}, relationships={})
    
    def process_query(self, query: str) -> QueryResult:
        """Process a query using the appropriate retriever with fallback mechanisms."""
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Check if the query is complex and needs decomposition
            if self.decomposer.is_complex_query(query):
                self.logger.info("Query is complex. Decomposing into sub-queries.")
                sub_queries = self.decomposer.decompose(query)
                self.logger.info(f"Decomposed sub-queries: {sub_queries}")
                
                # Process each sub-query
                results = []
                for sub_query in sub_queries:
                    result = self._process_sub_query(sub_query)
                    results.append((sub_query, result))  # Store sub-query and its result
                
                # Combine the results
                combined_content = self._format_results(results)
                return QueryResult(
                    content=combined_content,
                    confidence=min(r.confidence for _, r in results),
                    source='hybrid',
                    metadata={'sub_queries': len(sub_queries)}
                )
            else:
                # Process the query as a single unit
                self.logger.info("Query is simple. Processing as a single unit.")
                result = self._process_sub_query(query)
                return QueryResult(
                    content=result.content,
                    confidence=result.confidence,
                    source=result.source,
                    metadata={'sub_queries': 1}
                )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return QueryResult(
                content=f"An error occurred: {str(e)}",
                confidence=0.0,
                source='error',
                metadata={'error': str(e)}
            )
    
    def _process_sub_query(self, query: str) -> QueryResult:
        """Process a sub-query using the appropriate retriever."""
        try:
            self.logger.info(f"Processing sub-query: {query}")
            
            # Route the sub-query
            route, confidence = self.router.route_query(query)
            self.logger.info(f"Sub-query routed to {route} with confidence {confidence}")
            
            # Process with SQL retriever if confidence is high
            if route == 'sql' and confidence > self.config.SQL_CONFIDENCE_THRESHOLD:
                sql_result = self.sql_retriever.retrieve(query)
                if sql_result.confidence > 0:
                    return sql_result

                # Fallback to RAG if SQL fails
                self.logger.info("SQL retrieval failed, falling back to RAG")
                rag_result = self.rag_retriever.retrieve(query)
                if rag_result.confidence > 0:
                    return rag_result

                # If both fail, return "I don't know"
                return QueryResult(
                    content="I don't know.",
                    confidence=0.0,
                    source='none',
                    metadata={'error': 'No relevant information found'}
                )

            # Process with RAG retriever if SQL is not confident
            rag_result = self.rag_retriever.retrieve(query)
            if rag_result.confidence > 0:
                return rag_result

            # If RAG fails, return "I don't know"
            return QueryResult(
                content="I don't know.",
                confidence=0.0,
                source='none',
                metadata={'error': 'No relevant information found'}
            )

        except Exception as e:
            self.logger.error(f"Error processing sub-query: {e}")
            return QueryResult(
                content=f"An error occurred: {str(e)}",
                confidence=0.0,
                source='error',
                metadata={'error': str(e)}
            )
    
    def _format_results(self, results: List[Tuple[str, QueryResult]]) -> str:
        """Format the results to display sub-queries and their answers."""
        formatted_results = []
        for sub_query, result in results:
            if result.confidence > 0:  # Only include confident results
                formatted_results.append(
                    f"Sub-query: {sub_query}\nAnswer: {result.content}\n"
                )
            else:
                formatted_results.append(
                    f"Sub-query: {sub_query}\nAnswer: I don't know.\n"
                )
        return "\n".join(formatted_results)

###################
# Example Usage
###################

if __name__ == "__main__":
    # Initialize the agent
    config = Config()
    agent = HybridAgent(
        database_uri="postgresql://username:password@localhost:5432/database",
        docs_directory="/docs",
        config=config
    )
    
    # Test queries
    test_queries = [
        "Who is Ilya Sutskever, What is the revenue in the year 2021, and Who is the artist for Walking Into Clarksdale?",
        "List the top 5 customers who have spent the most, including their total spending, and the genres of music they purchased most frequently?"
    ]
    
    for query in test_queries:
        result = agent.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Source: {result.source}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response:\n{result.content}")
