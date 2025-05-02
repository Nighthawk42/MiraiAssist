 ## MiraiAssist v0.0.1-RAG

 A modular Python voice/text assistant framework featuring RAG (Retrieval-Augmented Generation) for long-term conversation memory, real-time STT/TTS, and a customizable UI.

## Project Status
This project is very much in the alpha state. It works. But it still has quirks and improvement. Especially in the GUI department. I have little to no experience designing nice GUIs.

## TODO

Add a hot-word like "Hey Mirai".

Design a better GUI. Maybe use a TUI instead?

Move everything to an embedded Python.

 ## Features

 *   **Voice & Text Input:** Interact via Push-to-Talk, a Record button, or a text input box.
 *   **Speech-to-Text (STT):** Uses `faster-whisper` for efficient local transcription with VAD support.
 *   **LLM Interaction:** Connects to any OpenAI-compatible API (like LM Studio, Ollama, vLLM, or OpenAI) for generating responses. Supports streaming output.
 *   **Retrieval-Augmented Generation (RAG):**
     *   Overcomes basic context window limitations by indexing the entire conversation history.
     *   Uses `sentence-transformers` for semantic embeddings and `ChromaDB` for efficient vector storage and retrieval.
     *   Augments LLM prompts with relevant historical context, enabling better long-term memory recall.
 *   **Text-to-Speech (TTS):** Uses `Kokoro` for generating speech output locally.
 *   **Modular Design:** Core functionalities (Audio, STT, LLM, Context, TTS, UI, Config, System) are separated into manageable modules.
 *   **Configurable:** Most settings managed via `config.yaml` (API endpoints, models, devices, RAG parameters, PTT keys, logging).
 *   **Graphical User Interface (UI):** Built with `customtkinter`, providing a themeable interface for conversation history, logs, and interaction.
 *   **Enhanced Logging:** Uses `rich` for formatted and colorful console output (configurable). Logs are also saved to files.

 ## Technology Stack

 *   **Python:** 3.9+
 *   **UI:** CustomTkinter
 *   **Audio:** PyAudio, NumPy, SoundFile
 *   **STT:** faster-whisper
 *   **TTS:** Kokoro
 *   **LLM Client:** openai (v1.0+)
 *   **RAG:** sentence-transformers, chromadb, tiktoken
 *   **Configuration:** PyYAML
 *   **Logging:** Rich (optional console), standard logging
 *   **Packaging/Dependencies:** uv (recommended) or pip

 ## Getting Started

 ### Prerequisites

 *   **Python:** Version 3.9 or higher recommended.
 *   **Git:** For cloning the repository.
 *   **C++ Build Tools (Windows):** May be required for dependencies like `PyAudio` or potentially `ChromaDB`'s C++ components. Install "Microsoft C++ Build Tools" via the Visual Studio Installer.
 *   **PortAudio (Linux/macOS):** Required by `PyAudio`. Install via your system's package manager (e.g., `sudo apt-get install portaudio19-dev` on Debian/Ubuntu, `brew install portaudio` on macOS).
 *   **(Optional) NVIDIA GPU + CUDA:** Required if you want to run STT or embedding models on the GPU (`device: cuda` in `config.yaml`). Ensure compatible CUDA Toolkit and PyTorch with CUDA support are installed.

 ### Installation

 1.  **Clone the repository:**
     ```bash
     git clone https://github.com/YourUsername/MiraiAssist.git  Replace with your repo URL
     cd MiraiAssist
     ```

 2.  **Create and activate a virtual environment:**
     ```bash
      Using Python's built-in venv
     python -m venv .venv

      Activate (Windows PowerShell)
     .\.venv\Scripts\Activate.ps1
      Activate (Windows CMD)
     .\.venv\Scripts\activate.bat
      Activate (Linux/macOS)
     source .venv/bin/activate
     ```

 3.  **Install dependencies using `uv` (recommended) or `pip`:**
     ```bash
      Using uv (Faster) - Ensure torch URL matches your CUDA/CPU needs
     uv pip install customtkinter pyaudio numpy soundfile pyyaml rich "faster-whisper @ git+https://github.com/SYSTRAN/faster-whisper.git" openai kokoro sentence-transformers chromadb tiktoken torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

      --- OR ---

      Using pip (Create requirements.txt first or install directly)
      Example direct install (adjust torch index URL if needed):
      pip install customtkinter pyaudio numpy soundfile pyyaml rich "faster-whisper @ git+https://github.com/SYSTRAN/faster-whisper.git" openai kokoro sentence-transformers chromadb tiktoken torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
     ```
     *Note: Adjust the torch `--extra-index-url` based on your specific OS and CUDA version. Visit [pytorch.org](https://pytorch.org/) for the correct command.*
     *Note: `faster-whisper` is installed directly from GitHub here. You can also try `pip install faster-whisper`.*

 ### Configuration

 1.  **Copy `config.yaml`:** If `config.yaml` doesn't exist, use the example content from the repository/documentation.

 2.  **Edit `config.yaml`:** Open the file and configure these critical settings:
     *   **`llm.api_base_url`:** URL for your OpenAI-compatible LLM server.
     *   **`llm.model_name`:** Model identifier your LLM server expects.
     *   **`llm.api_key_env_var`:** Name of the environment variable holding your API key (e.g., `OPENAI_API_KEY`) or `NONE` if no key needed. Ensure the variable is set *before* running.
     *   **`llm.model_context_window`:** **Crucial!** Set this to the max token limit of your LLM (e.g., 8192 for Llama 3 8B). Enables prompt length checks. Set to 0 or remove to disable checks.
     *   **`stt.model_size`:** `faster-whisper` model size (e.g., `base.en`).
     *   **`stt.device` / `stt.compute_type`:** Configure for CPU or CUDA GPU.
     *   **`context_manager.embedding_model_name`:** `sentence-transformers` model.
     *   (Optional) `audio.*`, `tts.*`, `logging.*`, `activation.push_to_talk_key`.

 3.  **First Run Model Downloads:** STT, TTS, and embedding models may download on first launch.

 ## Running the Application

 Ensure your virtual environment is activated and you are in the project's root directory.

 ```bash
 python main.py
 ```

 ## Usage

 *   **Voice Input:** Use PTT key (default `Ctrl+Space`) or the Record/Stop Recording button.
 *   **Text Input:** Type in the bottom text box. Press `Enter` or click "Send". `Shift+Enter` for newlines.
 *   **Theme:** Options > Theme menu.
 *   **Clear History:** Options > Clear Conversation History (erases memory and RAG index).

 ## Directory Structure

 ```
 MiraiAssist/
 ├── .venv/
 ├── data/
 │   ├── conversation_state.json  Full raw chat history
 │   └── chroma_db/             Persistent vector store
 ├── logs/
 ├── modules/
 │   ├── __init__.py
 │   ├── audio_manager.py
 │   ├── config_manager.py
 │   ├── context_manager.py   RAG implementation
 │   ├── llm_manager.py
 │   ├── stt_manager.py
 │   ├── system_manager.py
 │   ├── tts_manager.py
 │   └── ui_manager.py
 ├── placeholder.png          Replace with your screenshot
 ├── config.yaml
 ├── main.py
 └── README.md
 ```

 ## Acknowledgements

 *   CustomTkinter
 *   PyAudio
 *   Faster Whisper (Systran)
 *   Kokoro TTS (hexgrad)
 *   Sentence Transformers (UKPLab)
 *   ChromaDB
 *   Tiktoken (OpenAI)
 *   Rich (Textualize)
 *   OpenAI Python Client
 *   Hugging Face Hub & Transformers
