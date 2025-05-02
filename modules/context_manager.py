# ================================================
# FILE: modules/context_manager.py
# ================================================

import json
import logging
from pathlib import Path
import time
import shutil
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Dummy class for type hinting if needed, error raised in init
    class SentenceTransformer: pass

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings # Use specific Settings import
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    # Dummy classes/module
    class chromadb:
         @staticmethod
         def PersistentClient(*args, **kwargs): pass
         class Collection: pass
    class ChromaSettings: pass


# Use relative import for ConfigManager
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ContextManagerError(Exception):
    """Custom exception for ContextManager specific errors."""
    pass

class ContextManager:
    """
    Manages conversation history using RAG.

    - Stores full history in JSON.
    - Indexes messages into a ChromaDB vector store.
    - Retrieves relevant past messages based on user queries.
    """
    DEFAULT_STORAGE_PATH = "data/conversation_state.json"
    DEFAULT_VECTOR_DB_PATH = "data/chroma_db"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_COLLECTION_NAME = "mirei_chat_history"
    DEFAULT_RETRIEVAL_RESULTS = 3
    DEFAULT_INCLUDE_RECENT = 2

    def __init__(self, config: ConfigManager):
        """Initializes RAG components and loads history."""
        logger.info("Initializing ContextManager (RAG)...")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ContextManagerError("Required library 'sentence-transformers' not installed. Run: uv add sentence-transformers")
        if not CHROMA_AVAILABLE:
             raise ContextManagerError("Required library 'chromadb' not installed. Run: uv add chromadb")

        cfg_section = config.get("context_manager", default={})

        self.storage_path: Path = Path(cfg_section.get("storage_path", self.DEFAULT_STORAGE_PATH)).resolve()
        self.vector_db_path: Path = Path(cfg_section.get("vector_db_path", self.DEFAULT_VECTOR_DB_PATH)).resolve()
        self.embedding_model_name: str = cfg_section.get("embedding_model_name", self.DEFAULT_EMBEDDING_MODEL)
        self.collection_name: str = cfg_section.get("collection_name", self.DEFAULT_COLLECTION_NAME)
        self.n_retrieval_results: int = int(cfg_section.get("retrieval_results", self.DEFAULT_RETRIEVAL_RESULTS))
        self.n_include_recent: int = int(cfg_section.get("include_recent_messages", self.DEFAULT_INCLUDE_RECENT))

        self.messages: List[Dict[str, str]] = []
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None # Use ClientAPI type hint
        self.collection: Optional[chromadb.Collection] = None

        # 1. Load Embedding Model
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load SentenceTransformer model '{self.embedding_model_name}': {e}", exc_info=True)
            raise ContextManagerError(f"Failed to load embedding model: {e}") from e

        # 2. Initialize ChromaDB
        try:
            logger.info(f"Initializing ChromaDB client at: {self.vector_db_path}")
            # Ensure directory exists for persistent client
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=ChromaSettings(anonymized_telemetry=False) # Disable telemetry
            )
            # Get or create the collection
            logger.info(f"Getting or creating Chroma collection: {self.collection_name}")
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                # Optionally specify embedding function if not using default OpenAI
                # metadata={"hnsw:space": "cosine"} # Default is L2, cosine often better for ST
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ready. Item count: {self.collection.count()}")

        except Exception as e:
             logger.critical(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
             raise ContextManagerError(f"ChromaDB initialization failed: {e}") from e

        # 3. Load full history from JSON
        self._load_full_history()

        # 4. Index loaded history (can be slow for large histories on first run)
        self._initial_index()

        logger.info("ContextManager (RAG) initialized successfully.")

    def _load_full_history(self):
        """Loads the complete conversation history from JSON, backing up corrupted files."""
        # This function remains largely the same as the improved version from before
        # Just ensures self.messages holds the full history.
        if self.storage_path.exists() and self.storage_path.is_file():
            try:
                logger.info(f"Loading full conversation history from {self.storage_path}")
                with self.storage_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        logger.warning(f"History file {self.storage_path} is empty.")
                        self.messages = []
                        return
                    loaded_data = json.loads(content)

                if isinstance(loaded_data, list):
                    self.messages = [
                        msg for msg in loaded_data
                        if isinstance(msg, dict) and "role" in msg and "content" in msg
                    ]
                    logger.info(f"Loaded {len(self.messages)} messages from history file.")
                    if len(self.messages) != len(loaded_data):
                         logger.warning("Some invalid message formats found in history file were skipped.")
                else:
                    logger.warning(f"History file {self.storage_path} does not contain a list. Starting fresh.")
                    self.messages = []

            except (json.JSONDecodeError, IOError, Exception) as e:
                error_type = type(e).__name__
                logger.error(f"Failed to load/parse history file {self.storage_path} ({error_type}): {e}. Backing up and starting fresh.", exc_info=True)
                try:
                    backup_path = self.storage_path.with_name(
                        f"{self.storage_path.stem}_corrupted_{int(time.time())}{self.storage_path.suffix}"
                    )
                    shutil.move(str(self.storage_path), str(backup_path))
                    logger.info(f"Backed up corrupted history file to: {backup_path}")
                except Exception as backup_e:
                    logger.error(f"Failed to back up corrupted history file {self.storage_path}: {backup_e}", exc_info=True)
                self.messages = []
        else:
            logger.info(f"History file not found at {self.storage_path}. Starting with empty history.")
            self.messages = []

    def _initial_index(self):
        """Indexes messages from the loaded history if they aren't already in ChromaDB."""
        if not self.collection or not self.embedding_model:
            logger.error("Cannot perform initial index: Chroma collection or embedding model not available.")
            return

        logger.info("Performing initial check/indexing of loaded history...")
        start_time = time.time()
        added_count = 0
        existing_ids = set(self.collection.get(include=[])['ids']) # Efficient way to get all IDs

        ids_to_add = []
        embeddings_to_add = []
        documents_to_add = []
        metadatas_to_add = []

        for i, msg in enumerate(self.messages):
            msg_id = f"msg_{i}" # Simple index-based ID
            if msg_id not in existing_ids:
                content = msg.get("content", "")
                role = msg.get("role", "unknown")
                if content: # Only index messages with content
                    ids_to_add.append(msg_id)
                    # Embedding happens in batch later
                    documents_to_add.append(content)
                    metadatas_to_add.append({"role": role, "index": i})
                    added_count += 1

        # Batch embedding and adding
        if ids_to_add:
            logger.info(f"Found {added_count} messages from history to index...")
            try:
                # Calculate embeddings in batch
                embeddings_to_add = self.embedding_model.encode(documents_to_add, show_progress_bar=False).tolist()

                # Add to ChromaDB in batch
                self.collection.add(
                    ids=ids_to_add,
                    embeddings=embeddings_to_add,
                    documents=documents_to_add,
                    metadatas=metadatas_to_add
                )
                logger.info(f"Successfully indexed {added_count} messages.")
            except Exception as e:
                logger.error(f"Error during batch indexing: {e}", exc_info=True)
                # Potential issue: partial add? Chroma handles batches transactionally usually.
        else:
            logger.info("No new messages from loaded history needed indexing.")

        end_time = time.time()
        logger.info(f"Initial indexing check completed in {end_time - start_time:.2f} seconds.")

    def _index_message(self, msg_index: int, message: Dict[str, str]):
        """Adds a single message to the vector store."""
        if not self.collection or not self.embedding_model:
            logger.error("Cannot index message: Chroma collection or embedding model not available.")
            return

        msg_id = f"msg_{msg_index}"
        content = message.get("content", "")
        role = message.get("role", "unknown")

        if not content:
             logger.debug(f"Skipping indexing for message {msg_id} (no content).")
             return

        try:
            logger.debug(f"Indexing message: {msg_id} (Role: {role})")
            embedding = self.embedding_model.encode([content], show_progress_bar=False)[0].tolist()
            self.collection.add(
                ids=[msg_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"role": role, "index": msg_index}]
            )
        except Exception as e:
            logger.error(f"Failed to index message {msg_id}: {e}", exc_info=True)


    def add_message(self, role: str, content: str):
        """
        Adds a message to the in-memory history and indexes it in the vector store.
        Does NOT save the JSON file automatically.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid message role: '{role}'. Must be 'user' or 'assistant'.")
        if not isinstance(content, str):
             logger.warning(f"Message content is not a string (type: {type(content)}). Converting to string.")
             content = str(content)

        logger.debug(f"Adding message to memory - Role: {role}, Content: '{content[:50]}...'")
        new_message = {"role": role, "content": content}
        self.messages.append(new_message)
        new_message_index = len(self.messages) - 1

        # Index the new message immediately
        self._index_message(new_message_index, new_message)

        # NOTE: No condensation/truncation happens here anymore
        # NOTE: No automatic JSON save happens here anymore

    def retrieve_relevant_context(self, query: str) -> List[Dict[str, str]]:
        """Retrieves messages from history relevant to the query."""
        if not self.collection or not self.embedding_model:
             logger.error("Cannot retrieve context: Chroma collection or embedding model not available.")
             return []
        if not query:
             logger.warning("Cannot retrieve context for empty query.")
             return []
        if self.collection.count() == 0:
             logger.debug("Skipping retrieval: Vector store is empty.")
             return []


        try:
            logger.debug(f"Retrieving {self.n_retrieval_results} relevant messages for query: '{query[:60]}...'")
            start_time = time.time()

            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0].tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.n_retrieval_results, self.collection.count()), # Don't request more than available
                include=["documents", "metadatas", "distances"] # Include distance for potential filtering/logging
            )

            end_time = time.time()
            logger.debug(f"Retrieval query finished in {end_time - start_time:.3f} seconds.")

            # Process results
            retrieved_messages = []
            if results and results.get("ids") and results["ids"][0]: # Chroma returns lists within lists
                retrieved_ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                for i, doc_id in enumerate(retrieved_ids):
                     role = metadatas[i].get("role", "unknown")
                     content = documents[i]
                     distance = distances[i]
                     logger.debug(f"  Retrieved: ID={doc_id}, Role={role}, Distance={distance:.4f}, Content='{content[:50]}...'")
                     retrieved_messages.append({"role": role, "content": content})
            else:
                 logger.debug("No relevant messages found by Chroma query.")

            # Sort by original index? Chroma doesn't guarantee order, but similarity search is the primary goal.
            # If original order is desired *among retrieved items*, we'd need to sort by metadata['index'] here.
            # For now, return in similarity order as Chroma gives them.

            return retrieved_messages

        except Exception as e:
             logger.error(f"Error during context retrieval: {e}", exc_info=True)
             return [] # Return empty list on error

    def get_recent_messages(self, num_turns: int) -> List[Dict[str, str]]:
         """Gets the last N turns (user+assistant pairs) from history."""
         if num_turns <= 0:
             return []
         # A turn is typically user + assistant, so num_messages = num_turns * 2
         num_messages = num_turns * 2
         return self.messages[-num_messages:] # Slice the end of the list

    def save_context(self):
        """Saves the current full conversation history atomically to the JSON storage file."""
        # This function remains the same as the improved atomic save version
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        final_path = self.storage_path

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving full history ({len(self.messages)} messages) atomically to {final_path}")
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            shutil.move(str(temp_path), str(final_path))
            logger.info(f"Full history saved successfully to {final_path}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to write history file to {final_path} (or temp file {temp_path}): {e}", exc_info=True)
            if temp_path.exists():
                try: temp_path.unlink()
                except OSError: pass
        except Exception as e:
            logger.error(f"Unexpected error saving history to {final_path}: {e}", exc_info=True)
            if temp_path.exists():
                try: temp_path.unlink()
                except OSError: pass

    def clear_context(self):
        """Clears history in memory, clears the vector store, and saves the empty state."""
        logger.info("Clearing conversation context (memory, vector store, and file)...")
        self.messages = []

        # Clear the Chroma collection
        if self.collection:
            try:
                logger.warning(f"Deleting all items from Chroma collection: {self.collection_name}")
                existing_ids = self.collection.get(include=[])['ids']
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
                logger.info("Chroma collection cleared.")
            except Exception as e:
                logger.error(f"Failed to clear Chroma collection '{self.collection_name}': {e}", exc_info=True)
                # Continue with clearing memory and file even if DB clear fails

        # Save the empty context to file
        self.save_context()

    @property
    def history(self) -> List[Dict[str, str]]:
         """Provides read-only access to the full message history."""
         return list(self.messages) # Return a copy

    def shutdown(self):
         """Cleanly shuts down components (if necessary)."""
         # ChromaDB PersistentClient doesn't explicitly require shutdown usually,
         # but can be good practice if there were explicit connections.
         logger.info("ContextManager shutting down...")
         # Unload embedding model? Not strictly necessary unless memory is critical.
         self.embedding_model = None
         self.chroma_client = None # Clear references
         self.collection = None
         logger.info("ContextManager resources released.")