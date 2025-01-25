
## Overview

`Final_1.py` implements a Hybrid Agent System that combines SQL database querying with document retrieval capabilities. It utilizes various libraries and techniques to process queries, retrieve relevant information, and summarize results.

## Dependencies

The following libraries are required to run this code:

- `os`
- `requests`
- `logging`
- `re`
- `dataclasses`
- `functools`
- `typing`
- `datetime`
- `pathlib`
- `concurrent.futures`
- `langchain_community`
- `llama_index`
- `PyPDF2`
- `docx`

## Configuration

### Config Class

The `Config` class holds configuration settings for the Hybrid Agent system, including API keys, model settings, and thresholds for confidence levels.

```python
@dataclass
class Config:
    OPENAI_API_KEY: str
    DEEPSEEK_API_KEY: str
    LLM_MODEL: str
    TEMPERATURE: float
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    CACHE_SIZE: int
    SQL_CONFIDENCE_THRESHOLD: float
    RAG_CONFIDENCE_THRESHOLD: float
    DENSE_WEIGHT: float
    SPARSE_WEIGHT: float
    ENABLE_CACHE: bool
    CACHE_TTL: int
```

## Data Models

### QueryResult Class

Represents the result of a query, including content, confidence score, source, and metadata.

```python
@dataclass
class QueryResult:
    content: str
    confidence: float
    source: str
    metadata: Dict
    timestamp: datetime = datetime.now()
```

### SchemaInfo Class

Contains information about the database schema, including tables and relationships.

```python
@dataclass
class SchemaInfo:
    tables: Dict[str, List[str]]
    relationships: Dict[str, List[str]]
```

## Document Processing

### load_documents

Loads documents from a specified directory and splits them into smaller chunks for processing.

```python
def load_documents(docs_path: Path) -> List[Document]:
```

### load_pdf

Loads a PDF document and extracts its text content.

```python
def load_pdf(file_path: Path) -> Optional[Document]:
```

### load_docx

Loads a DOCX document and extracts its text content.

```python
def load_docx(file_path: Path) -> Optional[Document]:
```

## Retrievers

### SQLRetriever Class

Handles SQL database queries using the LangChain library.

```python
class SQLRetriever:
    def __init__(self, db: SQLDatabase, config: Config):
    def retrieve(self, query: str) -> QueryResult:
```

### RAGRetriever Class

Handles document retrieval using Retrieval-Augmented Generation (RAG) techniques.

```python
class RAGRetriever:
    def __init__(self, documents: List[Document], config: Config):
    def retrieve(self, query: str) -> QueryResult:
```

## Query Routing

### QueryRouter Class

Routes queries to the appropriate retrievers based on schema and query patterns.

```python
class QueryRouter:
    def __init__(self, schema_info: SchemaInfo, config: Config):
    def route_query(self, query: str) -> Tuple[str, float]:
```

## Query Decomposition

### QueryDecomposer Class

Decomposes complex queries into simpler sub-queries.

```python
class QueryDecomposer:
    def __init__(self, config: Config):
    def decompose(self, query: str) -> List[str]:
```

## Main Agent

### HybridAgent Class

The main class that combines SQL and RAG capabilities to process queries.

```python
class HybridAgent:
    def __init__(self, database_uri: str, docs_directory: str, config: Optional[Config] = None):
    def process_query(self, query: str) -> QueryResult:
```

## Example Usage

The following code demonstrates how to initialize the Hybrid Agent and process a query:

```python
if __name__ == "__main__":
    config = Config()
    agent = HybridAgent(
        database_uri="postgresql://username:password@localhost:5432/database",
        docs_directory="/path/to/documents",
        config=config
    )
    
    test_queries = [
        "Who is the artist for Balls to the Wall?",
    ]
    
    for query in test_queries:
        result = agent.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Source: {result.source}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response:\n{result.content}")
```

## Logging

The system uses logging to capture errors and important events. Logs are written to both a file (`hybrid_agent.log`) and the console.
