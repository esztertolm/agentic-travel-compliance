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
    Scans the data directory for PDF files and incrementally adds new documents
    to the local ChromaDB. It skips entire files if they are already present in the database.
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

    if force_recreate and os.path.exists(AppConfig.DB_DIR):
        logger.warning("Force recreate requested. Wiping existing vector database directory...")
        import shutil
        shutil.rmtree(AppConfig.DB_DIR, ignore_errors=True)

    logger.info("Loading/Initializing ChromaDB at '%s'...", AppConfig.DB_DIR)
    vector_db = Chroma(persist_directory=AppConfig.DB_DIR, embedding_function=embeddings)

    pdf_pattern = os.path.join(AppConfig.DATA_DIR, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        error_msg = f"No PDF files found in data directory: {AppConfig.DATA_DIR}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info("Found %d PDF file(s) in data folder.", len(pdf_files))

    new_chunks = []

    # Process each PDF file individually
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info("Checking document: %s", filename)
        
        try:
            existing_data = vector_db.get(where={"source_file": filename}, limit=1)
            if existing_data and existing_data.get("ids") and len(existing_data["ids"]) > 0:
                logger.info("Document '%s' is already in the database. Skipping entire file...", filename)
                continue
        except Exception as e:
            logger.warning("Could not check existence for %s: %s", filename, str(e))

        logger.info("New document detected: '%s'. Processing...", filename)
        
        try:
            loader = PyPDFLoader(pdf_path)
            loaded_pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=AppConfig.CHUNK_SIZE,
                chunk_overlap=AppConfig.CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            
            for page_num, page_doc in enumerate(loaded_pages):
                page_chunks = text_splitter.split_documents([page_doc])
                
                for chunk in page_chunks:
                    chunk.metadata["source_file"] = filename
                    chunk.metadata["page"] = page_num + 1
                    new_chunks.append(chunk)
                    
        except Exception as e:
            logger.error("Failed to process PDF document %s: %s", filename, str(e))
            continue

    if new_chunks:
        logger.info("Adding %d new text chunks to ChromaDB...", len(new_chunks))
        try:
            # We don't strictly need to provide manual IDs anymore, Chroma will generate UUIDs
            vector_db.add_documents(documents=new_chunks)
            logger.info("Incremental update complete. New chunks successfully persisted!")
        except Exception as e:
            logger.error("Failed to write new documents to vector store: %s", str(e))
            raise e
    else:
        logger.info("No new documents detected. Database is fully up-to-date!")

    return vector_db


def get_retriever():
    """
    Returns a retriever object configured for similarity search.
    """
    db = initialize_vector_db(force_recreate=False)
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": AppConfig.RETRIEVER_K}
    )


if __name__ == "__main__":
    from utils.logging import setup_logger
    setup_logger()

    logger.info("=== Starting Optimized Incremental Pipeline Test ===")
    try:
        db = initialize_vector_db(force_recreate=False)
        
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
        logger.info("=== Optimized Incremental Pipeline Test Complete ===")