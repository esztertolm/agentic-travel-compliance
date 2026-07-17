import os
import glob
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

from configuration.config import AppConfig

logger = logging.getLogger(__name__)


def initialize_vector_db(force_recreate: bool = False) -> Chroma:
    """
    Scans the data directory for all PDF files, loads them, splits them into chunks, 
    generates vector embeddings, and persists them into a local ChromaDB.
    
    If the database already exists and force_recreate is False, it loads the existing one.
    """
    logger.info("Initializing HuggingFaceEmbeddings with model: %s", AppConfig.EMBEDDING_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )
    except Exception as e:
        logger.error("Failed to load embedding model: %s", str(e))
        raise e

    # Check if DB already exists and load it to avoid redundant processing
    if os.path.exists(AppConfig.DB_DIR) and not force_recreate:
        logger.info("Existing vector database found at '%s'. Loading...", AppConfig.DB_DIR)
        return Chroma(persist_directory=AppConfig.DB_DIR, embedding_function=embeddings)

    logger.info("Starting fresh vector database pipeline...")
    
    # Find all PDFs in the data directory
    pdf_pattern = os.path.join(AppConfig.DATA_DIR, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        error_msg = f"No PDF files found in data directory: {AppConfig.DATA_DIR}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info("Found %d PDF file(s) to process: %s", len(pdf_files), [os.path.basename(f) for f in pdf_files])
    
    all_documents = []
    for pdf_path in pdf_files:
        logger.info("Loading document: %s", pdf_path)
        try:
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source_file"] = os.path.basename(pdf_path)
            all_documents.extend(loaded_docs)
        except Exception as e:
            logger.error("Failed to parse PDF document %s: %s", pdf_path, str(e))
            continue

    if not all_documents:
        raise ValueError("Failed to load any valid text documents from the specified PDFs.")

    # Split documents into manageable chunks using config settings
    logger.info("Splitting %d combined pages into chunks...", len(all_documents))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=AppConfig.CHUNK_SIZE,
        chunk_overlap=AppConfig.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(all_documents)
    logger.info("Successfully generated %d chunks from all documents.", len(chunks))

    # Vectorize and save to local disk
    logger.info("Embedding chunks and saving to ChromaDB at '%s'...", AppConfig.DB_DIR)
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=AppConfig.DB_DIR
        )
        logger.info("Vector database successfully built and persisted!")
        return vector_db
    except Exception as e:
        logger.error("Failed to write to vector store: %s", str(e))
        raise e


def get_retriever():
    """
    Returns a retriever object configured for similarity search.
    """
    db = initialize_vector_db()
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": AppConfig.RETRIEVER_K}
    )


if __name__ == "__main__":
    from utils.logging import setup_logger

    setup_logger()

    logger.info("=== Starting Multi-PDF Vector Database Pipeline Test ===")
    try:
        db = initialize_vector_db(force_recreate=True)
        
        test_query = "laundry services reimbursable"
        logger.info("Running similarity search test for query: '%s'", test_query)
        results = db.similarity_search(test_query, k=2)
        
        logger.info("Search finished. Number of results retrieved: %d", len(results))
        for i, doc in enumerate(results):
            logger.info(
                "Result %d (Source: %s, Page %s): %s...",
                i+1,
                doc.metadata.get('source_file', 'Unknown'),
                doc.metadata.get('page', 'N/A'),
                doc.page_content[:150].strip().replace('\n', ' ')
            )
            
    except Exception as ex:
        logger.critical("Pipeline test failed: %s", str(ex))
    finally:
        logger.info("=== Multi-PDF Vector Database Pipeline Test Complete ===")