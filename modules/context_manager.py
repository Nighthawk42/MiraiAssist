# modules/context_manager.py

import datetime
import json
import logging
from pathlib import Path
import time
import shutil
from typing import List, Dict, Any, Optional, Tuple

# --- Dependency Imports with Checks ---
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer: # Dummy for type hints
        def __init__(self, *args, **kwargs): pass
        def encode(self, *args, **kwargs): return []

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    class chromadb: # Dummy for type hints
         @staticmethod
         def PersistentClient(*args, **kwargs): return ChromaClientDummy()
    class ChromaClientDummy:
        def get_or_create_collection(self, *args, **kwargs): return ChromaCollectionDummy()
    class ChromaCollectionDummy:
        def count(self): return 0
        def get(self, *args, **kwargs): return {"ids": []}
        def add(self, *args, **kwargs): pass
        def query(self, *args, **kwargs): return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        def delete(self, *args, **kwargs): pass
    class ChromaSettings: pass


# Local Imports
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ContextManagerError(Exception):
    """Custom exception for ContextManager specific errors."""
    pass

class ContextManager:
    """
    Manages conversation history using Retrieval-Augmented Generation (RAG).
    - Stores the full, raw conversation history chronologically in a JSON file.
    - Indexes each message into a persistent ChromaDB vector store.
    - Provides methods to add messages and retrieve relevant past messages.
    """
    DEFAULT_STORAGE_PATH = "data/conversation_state.json"
    DEFAULT_VECTOR_DB_PATH = "data/chroma_db"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_COLLECTION_NAME = "mirei_chat_history"
    DEFAULT_RAG_N_RESULTS_FALLBACK = 3 # Fallback if n_results not provided to retrieve_relevant_context

    def __init__(self, config: ConfigManager):
        logger.info("Initializing ContextManager (RAG)...")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ContextManagerError(
                "Required library 'sentence-transformers' not installed. Run: uv add sentence-transformers"
            )
        if not CHROMA_AVAILABLE:
            raise ContextManagerError("Required library 'chromadb' not installed. Run: uv add chromadb")

        cfg_section = config.get("context_manager", default={})
        self.storage_path: Path = Path(cfg_section.get("storage_path", self.DEFAULT_STORAGE_PATH)).resolve()
        self.vector_db_path: Path = Path(cfg_section.get("vector_db_path", self.DEFAULT_VECTOR_DB_PATH)).resolve()
        self.embedding_model_name: str = cfg_section.get("embedding_model_name", self.DEFAULT_EMBEDDING_MODEL)
        self.collection_name: str = cfg_section.get("collection_name", self.DEFAULT_COLLECTION_NAME)
        # n_retrieval_results and n_include_recent from context_manager config are now primarily
        # used as fallbacks if methods are called without specific counts, or for direct use if any.
        # MemoryManager will use its own config for prompt construction.
        self.n_retrieval_results_fallback: int = int(cfg_section.get("retrieval_results", self.DEFAULT_RAG_N_RESULTS_FALLBACK))


        self.messages: List[Dict[str, Any]] = []
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.API] = None
        self.collection: Optional[chromadb.Collection] = None # Type hint for Chroma's Collection

        self._load_embedding_model()
        self._initialize_vector_db()
        self._load_full_history() # This now ensures 'original_index' is present
        self._synchronize_index()

        logger.info("ContextManager (RAG) initialized successfully.")

    def _load_embedding_model(self) -> None:
        logger.info(f"Loading embedding model: '{self.embedding_model_name}'...")
        start_time = time.time()
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            _ = self.embedding_model.encode(["test warm-up"], show_progress_bar=False) # Warm-up/check
            load_time = time.time() - start_time
            logger.info(f"Embedding model '{self.embedding_model_name}' loaded in {load_time:.2f}s.")
        except Exception as e:
            logger.critical(f"Failed to load SentenceTransformer model '{self.embedding_model_name}': {e}", exc_info=True)
            raise ContextManagerError(f"Embedding model load failed: {e}") from e

    def _initialize_vector_db(self) -> None:
        logger.info(f"Initializing ChromaDB client at: {self.vector_db_path}")
        try:
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                # metadata={"hnsw:space": "cosine"} # Optional: Explicitly set distance metric if needed
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ready. Initial item count: {self.collection.count()}")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise ContextManagerError(f"ChromaDB initialization failed: {e}") from e

    def _load_full_history(self) -> None:
        if self.storage_path.exists() and self.storage_path.is_file():
            try:
                logger.info(f"Loading full conversation history from {self.storage_path}")
                with self.storage_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    logger.warning(f"History file '{self.storage_path}' is empty.")
                    self.messages = []; return

                loaded_data = json.loads(content)
                if isinstance(loaded_data, list):
                    valid_messages = []
                    for i, msg_dict in enumerate(loaded_data):
                        if (isinstance(msg_dict, dict) and
                                "role" in msg_dict and isinstance(msg_dict["role"], str) and
                                "content" in msg_dict and # Allow empty content for system messages potentially
                                msg_dict["role"] in ["user", "assistant", "system"]): # Allow system role
                            # Ensure 'original_index' is present and correct
                            msg_copy = msg_dict.copy()
                            msg_copy['original_index'] = i # The index in the loaded list is its original_index
                            valid_messages.append(msg_copy)
                        else:
                            logger.warning(f"Skipping invalid message format at index {i} in history: {msg_dict}")
                    self.messages = valid_messages
                    logger.info(f"Loaded {len(self.messages)} valid messages from history.")
                else:
                    logger.warning(f"History file '{self.storage_path}' not a list. Starting fresh.")
                    self.messages = []
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load/parse history '{self.storage_path}' ({type(e).__name__}): {e}. Backing up.", exc_info=False)
                self._backup_corrupted_file(self.storage_path); self.messages = []
            except Exception as e:
                logger.error(f"Unexpected error loading history '{self.storage_path}': {e}. Backing up.", exc_info=True)
                self._backup_corrupted_file(self.storage_path); self.messages = []
        else:
            logger.info(f"History file not found at '{self.storage_path}'. Starting empty history.")
            self.messages = []

    def _backup_corrupted_file(self, file_path: Path) -> None:
        try:
            backup_dir = file_path.parent / "backups"; backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{file_path.stem}_corrupted_{timestamp}{file_path.suffix}"
            shutil.move(str(file_path), str(backup_path))
            logger.info(f"Backed up corrupted file to: {backup_path}")
        except Exception as backup_e:
            logger.error(f"Could not back up file '{file_path}': {backup_e}", exc_info=True)

    def _synchronize_index(self) -> None:
        if not self.collection or not self.embedding_model:
            logger.error("Cannot sync index: DB or model not ready."); return
        logger.info("Synchronizing vector index with loaded history...")
        start_time = time.time()
        messages_to_index: List[Dict[str, Any]] = [] # List of message dicts
        try:
            existing_db_ids_result = self.collection.get(include=[])
            existing_db_ids = set(existing_db_ids_result.get('ids', []))
            logger.debug(f"Found {len(existing_db_ids)} existing IDs in Chroma.")

            for msg in self.messages:
                # 'original_index' should have been set during _load_full_history or add_message
                msg_original_idx = msg.get('original_index')
                if msg_original_idx is None:
                    logger.error(f"Message found without original_index during sync: {str(msg)[:100]}. Skipping.")
                    continue
                msg_id = f"msg_{msg_original_idx}"
                if msg_id not in existing_db_ids:
                    messages_to_index.append(msg) # msg already contains 'original_index'

            if not messages_to_index: logger.info("Vector index is synchronized."); return
            logger.info(f"Found {len(messages_to_index)} messages from history to index...")

            ids_to_add, docs_to_add, metas_to_add = [], [], []
            for message in messages_to_index:
                content = message.get("content", "")
                role = message.get("role", "unknown")
                msg_original_idx = message['original_index'] # Should exist
                if content: # Only index messages with content
                    msg_id = f"msg_{msg_original_idx}"
                    ids_to_add.append(msg_id)
                    docs_to_add.append(content)
                    metas_to_add.append({"role": role, "original_index": msg_original_idx})

            if ids_to_add:
                logger.debug(f"Encoding {len(docs_to_add)} documents for batch indexing...")
                embeddings = self.embedding_model.encode(docs_to_add, show_progress_bar=False).tolist()
                logger.debug(f"Adding {len(ids_to_add)} items to ChromaDB...")
                self.collection.add(ids=ids_to_add, embeddings=embeddings, documents=docs_to_add, metadatas=metas_to_add)
                logger.info(f"Successfully indexed {len(ids_to_add)} messages from history.")
            else:
                 logger.info("No valid messages found to index after filtering.")
        except Exception as e:
            logger.error(f"Error during index synchronization: {e}", exc_info=True)
        finally:
             sync_time = time.time() - start_time
             logger.info(f"Index sync check completed in {sync_time:.2f}s.")

    def _index_message(self, msg_original_index: int, message: Dict[str, Any]) -> None:
        if not self.collection or not self.embedding_model:
            logger.error("Cannot index: DB or model not ready."); return
        msg_id = f"msg_{msg_original_index}"
        content = message.get("content", "")
        role = message.get("role", "unknown")
        if not content: logger.debug(f"Skipping indexing for msg_{msg_original_index} (no content)."); return
        try:
            embedding = self.embedding_model.encode([content], show_progress_bar=False)[0].tolist()
            self.collection.add(
                ids=[msg_id], embeddings=[embedding], documents=[content],
                metadatas=[{"role": role, "original_index": msg_original_index}]
            )
            logger.debug(f"Indexed message: {msg_id} (Role: {role}, OrigIdx: {msg_original_index})")
        except Exception as e:
            logger.error(f"Failed to index msg {msg_id}: {e}", exc_info=True)

    def add_message(self, role: str, content: str) -> None:
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: '{role}'. Must be 'user', 'assistant', or 'system'.")
        if not isinstance(content, str):
            content = str(content)

        new_message = {"role": role, "content": content}
        # Assign 'original_index' based on its future position in self.messages
        new_message['original_index'] = len(self.messages)
        self.messages.append(new_message)
        logger.debug(
            f"Added message to memory (OrigIdx: {new_message['original_index']}): Role={role}, Content='{content[:50]}...'"
        )
        self._index_message(new_message['original_index'], new_message)

    def retrieve_relevant_context(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.collection or not self.embedding_model:
            logger.error("Cannot retrieve: DB or model not ready."); return []
        if not query: logger.warning("Cannot retrieve context for empty query."); return []
        
        collection_count = self.collection.count()
        if collection_count == 0: logger.debug("Skipping retrieval: Vector store empty."); return []

        num_to_retrieve = n_results if n_results is not None else self.n_retrieval_results_fallback
        num_to_retrieve = min(num_to_retrieve, collection_count)
        if num_to_retrieve <= 0: logger.debug("No results to retrieve."); return []

        try:
            logger.info(f"Retrieving up to {num_to_retrieve} messages for query: '{query[:60]}...'")
            start_time = time.time()
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0].tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=num_to_retrieve,
                include=["documents", "metadatas", "distances"]
            )
            retrieval_time = time.time() - start_time
            logger.debug(f"ChromaDB query in {retrieval_time:.3f}s.")

            retrieved_messages = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    original_idx_val = metadata.get("original_index", -1)
                    try: original_idx = int(original_idx_val)
                    except (ValueError, TypeError): original_idx = -1; logger.warning(f"Invalid original_index {original_idx_val}")
                    
                    retrieved_messages.append({
                        "role": metadata.get("role", "unknown"),
                        "content": results["documents"][0][i] if results["documents"] and results["documents"][0] else "",
                        "metadata": {"original_index": original_idx, "distance": results["distances"][0][i] if results["distances"] and results["distances"][0] else float('inf')}
                    })
            logger.info(f"Retrieved {len(retrieved_messages)} relevant messages.")
            return retrieved_messages
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}", exc_info=True); return []

    def get_recent_messages(self, num_messages_to_fetch: int) -> List[Dict[str, Any]]:
        """Gets the last N messages. Ensures 'original_index' is present."""
        if num_messages_to_fetch <= 0: return []
        
        start_idx_slice = max(0, len(self.messages) - num_messages_to_fetch)
        recent_slice = self.messages[start_idx_slice:]

        # Ensure all messages in the slice have 'original_index'.
        # This primarily safeguards against older data formats if any were loaded
        # without 'original_index' (though _load_full_history attempts to add it).
        processed_recent: List[Dict[str, Any]] = []
        for i, msg_dict in enumerate(recent_slice):
            msg_copy = msg_dict.copy() # Work with a copy
            if 'original_index' not in msg_copy or not isinstance(msg_copy['original_index'], int):
                # Fallback: if somehow original_index is missing or invalid from the loaded message
                calculated_original_idx = start_idx_slice + i
                msg_copy['original_index'] = calculated_original_idx
                logger.warning(
                    f"ContextManager.get_recent_messages: Re-calculated missing/invalid 'original_index' "
                    f"({calculated_original_idx}) for recent message: {str(msg_dict.get('content',''))[:30]}..."
                )
            processed_recent.append(msg_copy)
        
        logger.debug(f"Retrieved {len(processed_recent)} messages ({num_messages_to_fetch} requested).")
        return processed_recent

    def save_context(self) -> None:
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving full history ({len(self.messages)} messages) to {self.storage_path}")
            # Save role, content, and original_index to allow reconstruction
            messages_to_save = [
                {"role": msg["role"], "content": msg["content"], "original_index": msg.get("original_index", idx)}
                for idx, msg in enumerate(self.messages)
            ]
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(messages_to_save, f, ensure_ascii=False, indent=2)
            shutil.move(str(temp_path), str(self.storage_path))
            logger.info("Full history saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save history '{self.storage_path}': {e}", exc_info=True)
            if temp_path.exists(): temp_path.unlink(missing_ok=True)

    def clear_context(self) -> None:
        logger.warning("Clearing conversation context (Memory, Vector Store, File)...")
        self.messages = []
        if self.collection:
            try:
                count = self.collection.count()
                if count > 0:
                    logger.info(f"Deleting {count} items from Chroma collection '{self.collection_name}'...")
                    # Efficient way to clear a Chroma collection (if API supports `delete_collection`)
                    # Or, if not, delete all items by IDs.
                    # For current chromadb versions, deleting by IDs is standard.
                    # If the collection can be deleted and recreated:
                    # self.chroma_client.delete_collection(name=self.collection_name)
                    # self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
                    # logger.info(f"Chroma collection '{self.collection_name}' deleted and recreated.")
                    # --- OR ---
                    all_ids_result = self.collection.get(include=[]) # Only need IDs
                    all_ids = all_ids_result.get('ids', [])
                    if all_ids:
                        self.collection.delete(ids=all_ids)
                        logger.info(f"Deleted {len(all_ids)} items from Chroma collection.")
                    else:
                         logger.info("Chroma collection was already empty (no IDs to delete).")
                else:
                    logger.info("Chroma collection already empty.")
            except Exception as e:
                logger.error(f"Failed to clear ChromaDB collection: {e}", exc_info=True)
        self.save_context() # Save the empty state
        logger.warning("Conversation context cleared.")

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self.messages) # Return a shallow copy

    def shutdown(self) -> None:
        logger.info("ContextManager shutting down...")
        self.embedding_model = None
        self.chroma_client = None # Chroma client usually handles its own persistence
        self.collection = None
        logger.info("ContextManager shutdown complete.")