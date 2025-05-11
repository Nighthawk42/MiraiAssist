# modules/memory_manager.py

import logging
from typing import List, Dict, Any, Optional

# Local Imports
from .config_manager import ConfigManager
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

class MemoryManagerError(Exception):
    """Custom exception for MemoryManager specific errors."""
    pass

class MemoryManager:
    """
    Manages the construction of conversation context for the LLM,
    orchestrating short-term (recent) and long-term (RAG) memory
    retrieval from the ContextManager.
    """
    DEFAULT_STM_WINDOW_TURNS = 2
    DEFAULT_LTM_RETRIEVAL_COUNT = 3

    def __init__(self, config: ConfigManager, context_manager: ContextManager):
        logger.info("Initializing MemoryManager...")
        if not isinstance(config, ConfigManager):
            raise MemoryManagerError("Invalid ConfigManager instance provided to MemoryManager.")
        if not isinstance(context_manager, ContextManager):
            raise MemoryManagerError("Invalid ContextManager instance provided to MemoryManager.")

        self.config = config
        self.context_manager = context_manager

        mem_cfg = self.config.get("memory_manager", default={})
        self.short_term_window_turns: int = int(mem_cfg.get(
            "short_term_window_turns", self.DEFAULT_STM_WINDOW_TURNS
        ))
        self.long_term_retrieval_count: int = int(mem_cfg.get(
            "long_term_retrieval_count", self.DEFAULT_LTM_RETRIEVAL_COUNT
        ))

        if self.short_term_window_turns < 0:
            logger.warning(f"MemoryManager 'short_term_window_turns' ({self.short_term_window_turns}) cannot be negative. Setting to 0.")
            self.short_term_window_turns = 0
        if self.long_term_retrieval_count < 0:
            logger.warning(f"MemoryManager 'long_term_retrieval_count' ({self.long_term_retrieval_count}) cannot be negative. Setting to 0.")
            self.long_term_retrieval_count = 0

        logger.info(
            f"MemoryManager configured: STM Turns={self.short_term_window_turns}, "
            f"LTM Count={self.long_term_retrieval_count}"
        )
        logger.info("MemoryManager initialized successfully.")

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the underlying ContextManager, which handles
        in-memory history and vector indexing.
        """
        try:
            self.context_manager.add_message(role, content)
            logger.debug(f"MemoryManager: Message (Role: {role}) passed to ContextManager.")
        except Exception as e:
            logger.error(f"MemoryManager: Error adding message via ContextManager: {e}", exc_info=True)

    def construct_prompt_context(self, current_query: str) -> List[Dict[str, str]]:
        """
        Constructs a list of context messages for the LLM prompt.
        The returned context should be *prior to* the current_query, as
        LLMManager will append the current_query.
        It attempts to enforce alternating user/assistant roles.
        """
        logger.debug(f"Constructing prompt context leading up to query: '{current_query[:50]}...'")

        # 1. Retrieve Long-Term Memory (RAG)
        retrieved_ltm: List[Dict[str, Any]] = []
        if self.long_term_retrieval_count > 0:
            try:
                retrieved_ltm = self.context_manager.retrieve_relevant_context(
                    query=current_query,
                    n_results=self.long_term_retrieval_count
                )
                logger.debug(f"Retrieved {len(retrieved_ltm)} LTM messages via RAG.")
            except Exception as e:
                logger.error(f"Error retrieving LTM from ContextManager: {e}", exc_info=True)
        else:
            logger.debug("LTM retrieval skipped (long_term_retrieval_count is 0).")

        # 2. Retrieve Short-Term Memory (Recent Messages)
        num_recent_messages_to_fetch = self.short_term_window_turns * 2
        recent_stm: List[Dict[str, Any]] = []
        if num_recent_messages_to_fetch > 0:
            try:
                # This fetches messages from history which now includes the current_query
                recent_stm = self.context_manager.get_recent_messages(num_recent_messages_to_fetch)
                logger.debug(f"Retrieved {len(recent_stm)} candidate STM messages (recent).")
            except Exception as e:
                logger.error(f"Error retrieving STM from ContextManager: {e}", exc_info=True)
        else:
            logger.debug("STM retrieval skipped (short_term_window_turns is 0).")

        # 3. Combine and initially sort all candidate messages by original_index
        combined_candidates_dict: Dict[int, Dict[str, str]] = {}
        current_query_original_index: Optional[int] = None

        # Determine original_index of current_query (it's the last in context_manager.messages)
        if self.context_manager.messages:
            last_message_in_full_history = self.context_manager.messages[-1]
            if last_message_in_full_history.get("role") == "user" and \
               last_message_in_full_history.get("content") == current_query:
                current_query_original_index = last_message_in_full_history.get("original_index")
                logger.debug(f"Identified current_query to exclude with original_index: {current_query_original_index}")

        for msg_source_name, msg_source_list in [("LTM", retrieved_ltm), ("STM", recent_stm)]:
            for msg in msg_source_list:
                original_index = msg.get("metadata", {}).get("original_index") if msg_source_name == "LTM" else msg.get("original_index")

                if original_index is not None and isinstance(original_index, int):
                    # Exclude the current_query itself from the context being built
                    if current_query_original_index is not None and original_index == current_query_original_index:
                        logger.debug(f"Skipping current_query (OrigIdx: {original_index}) from {msg_source_name} during initial assembly.")
                        continue
                    
                    content = msg.get("content", "").strip()
                    role = msg.get("role", "unknown")
                    if content and role in ["user", "assistant"]: # Only consider valid roles and non-empty content
                        combined_candidates_dict[original_index] = {"role": role, "content": content}
                else:
                    logger.warning(f"{msg_source_name} message missing valid 'original_index': {msg.get('content', '')[:30]}...")
        
        sorted_indices = sorted(combined_candidates_dict.keys())
        chronological_context: List[Dict[str, str]] = [
            combined_candidates_dict[idx] for idx in sorted_indices
        ]
        logger.debug(f"Assembled {len(chronological_context)} chronological context candidates (pre-alternation).")

        # 4. Enforce alternating roles to build final_context_messages
        final_context_messages: List[Dict[str, str]] = []
        last_added_role: Optional[str] = None

        for msg in chronological_context:
            current_role = msg["role"] # role should be "user" or "assistant" at this point
            current_content = msg["content"] # content should be non-empty and stripped

            if not final_context_messages: # First message to add to context
                # The very first message in history (after system prompt, handled by LLMManager)
                # ideally should be a 'user' message for most models.
                # However, if RAG pulls an 'assistant' message as the oldest relevant,
                # and there's no preceding 'user' message in `chronological_context`,
                # we might have an issue.
                # For now, let's just add the first valid message.
                final_context_messages.append(msg)
                last_added_role = current_role
                logger.debug(f"Alternation: Adding first message to context: Role={current_role}, Content='{current_content[:30]}...'")
            elif current_role != last_added_role:
                final_context_messages.append(msg)
                last_added_role = current_role
                logger.debug(f"Alternation: Adding message (role changed): Role={current_role}, Content='{current_content[:30]}...'")
            else: # Roles are the same as the last added message
                if current_role == "user":
                    # Merge with the previous user message
                    final_context_messages[-1]["content"] = (final_context_messages[-1]["content"] + "\n" + current_content).strip()
                    logger.debug(f"Alternation: Merged user message. New combined content starts: '{final_context_messages[-1]['content'][:30]}...'")
                elif current_role == "assistant":
                    # Replace the previous assistant message with this (presumably more relevant or later chronological) one
                    logger.debug(f"Alternation: Replacing previous assistant message ('{final_context_messages[-1]['content'][:30]}...') with new one ('{current_content[:30]}...').")
                    final_context_messages[-1] = msg
                    # last_added_role remains "assistant"
        
        # Final check: The context being returned to LLMManager should not cause an
        # [System, Assistant, User (current_query)] sequence if the history is short
        # and only an assistant message was selected for context.
        # If the very first message of our context is "assistant", and there's nothing before it,
        # it means the LLM prompt will be System, Assistant, User(current). This is often bad.
        # So, if `final_context_messages` has only one message and it's an assistant, we might clear it,
        # or if it starts with assistant and the *overall true history* implies a user should have come before it.
        # This specific edge case (first message in context being assistant) is what was causing the issue.
        # LLMManager adds System then current User. Context goes in between.
        # Prompt: System, [Context Messages], User (current)
        # If Context Messages = [Assistant, User, Assistant]
        # Result: System, Assistant, User, Assistant, User ( PROBLEM: S, A)
        if final_context_messages and final_context_messages[0].get("role") == "assistant":
            # If the very first message in our constructed context is 'assistant',
            # it will directly follow the 'system' prompt if no other 'user' message
            # precedes it from an earlier part of history not included in this RAG/STM window.
            # This is a common cause for the alternation error.
            # We remove this leading assistant message to allow the subsequent 'user' (current_query)
            # to follow the system prompt, or to allow a 'user' message later in final_context_messages
            # to be the first non-system message.
            logger.warning(
                f"Alternation: First message in constructed context is 'assistant' ('{final_context_messages[0]['content'][:30]}...'). "
                "Removing it to prevent System-Assistant start for the LLM."
            )
            final_context_messages.pop(0)
            
            # After removing, if the new first message is same role as next, re-evaluate (simple fix)
            if len(final_context_messages) >= 2 and final_context_messages[0].get("role") == final_context_messages[1].get("role"):
                logger.debug("Post-pop alternation check: Consecutive roles found after removing leading assistant.")
                if final_context_messages[0].get("role") == "user": # Two users
                    merged_user_content = (final_context_messages[0]["content"] + "\n" + final_context_messages[1]["content"]).strip()
                    final_context_messages[0]["content"] = merged_user_content
                    final_context_messages.pop(1)
                    logger.debug("Merged consecutive users after pop.")
                # Not typically expecting two assistants after pop, but could be added if needed.


        logger.info(f"MemoryManager returning final alternating prompt context with {len(final_context_messages)} messages.")
        return final_context_messages

    def clear_memory(self) -> None:
        logger.info("MemoryManager: Clearing all memory via ContextManager.")
        try:
            self.context_manager.clear_context()
        except Exception as e:
            logger.error(f"MemoryManager: Error clearing memory via ContextManager: {e}", exc_info=True)

    def get_full_history(self) -> List[Dict[str, Any]]:
        try:
            return self.context_manager.history
        except Exception as e:
            logger.error(f"MemoryManager: Error retrieving full history from ContextManager: {e}", exc_info=True)
            return []

    def shutdown(self) -> None:
        logger.info("MemoryManager shutting down...")
        logger.info("MemoryManager shutdown complete.")