---
title: ScholarAI
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ScholarAI — RAG Research Assistant

An intelligent research assistant powered by **Retrieval-Augmented Generation (RAG)**. Upload documents (PDF, TXT, MD), ask questions, and get answers grounded in your own knowledge base with source citations.

Built with **FastAPI**, **LangChain LCEL**, **ChromaDB**, and **Groq LLMs**.

---

## Features

- **📄 Document Ingestion** — Upload PDF, TXT, and Markdown files. Automatic chunking, embedding, and indexing into ChromaDB.
- **💬 RAG-Powered Chat** — Ask questions and get answers grounded in your uploaded documents. Session-based chat history with source citations.
- **🔐 User Authentication** — JWT-based signup/login with bcrypt password hashing.
- **⚙️ Settings Panel** — Choose from multiple Groq models, adjust temperature/top-p/max-tokens, configure API keys (Groq, OpenAI, Anthropic), set custom system prompts.
- **🧠 Memory & Notes** — Save research concepts, ideas, and notes with categories.
- **🌗 Light/Dark Theme** — Toggle between light and dark mode.
- **📱 Responsive UI** — Works on desktop and mobile.
- **🔌 API-First** — All functionality exposed via REST API for easy integration.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **AI/LLM** | Groq API (Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B) |
| **RAG Pipeline** | LangChain LCEL, ChromaDB vector store |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace, local) |
| **Auth** | JWT (python-jose), bcrypt (passlib) |
| **Frontend** | HTML, Tailwind CSS, Google Fonts (EB Garamond + Manrope) |
| **Deployment** | Render (render.yaml included) |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/abdulsamiuthwal-eng/ScholarAI.git
cd ScholarAI
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
AETHER_SECRET=your_jwt_secret_key
```

> Get a Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the Server

```bash
python backend.py
```

Or with uvicorn directly:

```bash
uvicorn backend:app --host 0.0.0.0 --port 8001 --reload
```

Open **http://localhost:8001** in your browser.

---

## API Endpoints

### Auth
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/auth/register` | Create account |
| `POST` | `/api/auth/login` | Sign in |
| `GET` | `/api/auth/me` | Get current user |

### Chat
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Send message (RAG) |
| `GET` | `/api/chat/{session_id}/history` | Get chat history |
| `DELETE` | `/api/chat/{session_id}` | Clear session |

### Documents
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/documents/upload` | Upload PDF/TXT/MD |
| `GET` | `/api/documents` | List indexed files |
| `DELETE` | `/api/documents/{filename}` | Delete file & index |

### Settings
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/settings` | Get settings (keys masked) |
| `PUT` | `/api/settings` | Update settings |
| `POST` | `/api/settings/test` | Test API connectivity |

### Memory
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/memory` | List saved entries |
| `POST` | `/api/memory` | Save a note/concept |
| `DELETE` | `/api/memory/{id}` | Delete entry |
| `DELETE` | `/api/memory` | Clear all memory |

### Models
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/models` | List available LLM models |

### Status
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | System status & stats |

---

## Deployment (Render)

1. Push to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Connect your repo.
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `GROQ_API_KEY`
   - `AETHER_SECRET` (auto-generated)

Or use the included `render.yaml` for **Infrastructure as Code** deployment.

---

## Project Structure

```
ScholarAI/
├── backend.py               # FastAPI app (routes, RAG pipeline, auth)
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment config
├── .env.example             # Environment variable template
├── .gitignore
├── public/
│   ├── index.html           # Main chat UI
│   └── login.html           # Auth/sign-in UI
├── chroma_db/               # Vector store (auto-created)
├── uploads/                 # Uploaded documents (auto-created)
└── .stitch/                 # Design system files (internal)
```

---

## License

MIT