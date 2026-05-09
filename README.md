# 🌲 Vectorless RAG — LLM Tree Search with PageIndex

[![Python](https://img.shields.io/badge/Python-3.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org)
[![PageIndex](https://img.shields.io/badge/PageIndex-Tree_Index-6C3DF4?style=for-the-badge)](https://pageindex.ai)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A production-ready **Vectorless RAG** implementation using PageIndex — replacing vector similarity search with **LLM reasoning over hierarchical document trees**. No vector database. No chunking. No cosine similarity. Just structured reasoning.

---

## 🔑 The Core Idea

Traditional RAG has a fundamental flaw:

```
Similarity ≠ Relevance
```

A chunk about "market conditions" may score higher in cosine similarity than the actual answer section — just because it shares more words with your query. This leads to hallucinations from irrelevant context.

**Vectorless RAG solves this:**

```
Traditional RAG:  query → embed → cosine_similarity → top-k chunks → answer
                  ❌ Finds what's SIMILAR, not what's RELEVANT

Vectorless RAG:   query + tree → LLM reasons → exact sections → answer  
                  ✅ Finds what ANSWERS the question
```

---

## 📊 Performance

| Metric | Traditional Vector RAG | Vectorless RAG (PageIndex) |
|---|---|---|
| FinanceBench accuracy | ~80% | **98.7%** |
| Hallucination source | Irrelevant chunks | None (LLM rejects poor context) |
| Retrieval explainability | ❌ Opaque score | ✅ Full reasoning trace |
| Infrastructure needed | Vector DB | JSON file |
| Domain expertise injection | Fine-tuning required | Prompt engineering only |

---

## 🏗️ Architecture

```
PDF Document
     │
     ▼
┌────────────────────────────────────┐
│         PageIndex Tree Builder     │
│                                    │
│  Reads document structure → builds │
│  hierarchical tree (like smart TOC)│
│                                    │
│  Document                          │
│  ├── Introduction (p.1-3)          │
│  ├── Methodology (p.4-12)          │
│  │   ├── Data Collection (p.4-7)   │
│  │   └── Analysis (p.8-12)         │
│  ├── Results (p.13-28)             │
│  └── Conclusion (p.29-32)          │
└────────────────┬───────────────────┘
                 │
                 ▼
         User Query comes in
                 │
                 ▼
┌────────────────────────────────────┐
│        LLM Tree Search             │
│                                    │
│  LLM reads query + full tree       │
│  Reasons: "Results section (p.13)  │
│  and Conclusion (p.29) contain     │
│  the answer to this query"         │
│  Returns: ["node_0013", "node_0029"]│
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│       Content Retrieval            │
│                                    │
│  Fetches full text of selected     │
│  nodes — exact sections, with      │
│  titles and page numbers           │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│        Answer Generation           │
│                                    │
│  LLM generates grounded answer     │
│  citing: section title + page num  │
│  No hallucination from bad context │
└────────────────────────────────────┘
```

---

## 📁 Notebook Structure

The notebook (`Vectorless_RAG_PageIndex.ipynb`) contains **24 code-only cells** with no explanation markup — clean, runnable, production-focused:

| Cells | What it does |
|---|---|
| 1–3 | Install, load API keys, initialize clients |
| 4–5 | Upload PDF, poll until tree index is ready |
| 6–8 | Fetch tree, visualize structure, count nodes |
| 9–10 | `llm_tree_search()` — core retrieval function + test |
| 11–12 | `fetch_node_content()` + `generate_answer()` |
| 13–14 | `vectorless_rag()` — full end-to-end pipeline + run |
| 15–17 | Expert-guided retrieval with domain rules |
| 18–19 | Multi-turn Chat API with conversation history |
| 20–23 | Self-hosted open-source mode (local, no cloud) |
| 24 | Cleanup — delete document from cloud |

---

## ⚙️ Core Functions

### `llm_tree_search(query, tree)`
Sends the query and compressed document tree to GPT-4o. The LLM reasons step-by-step over the tree structure and returns the node IDs most likely to contain the answer.

```python
result = llm_tree_search("What are the key findings?", pageindex_tree)
# Returns:
# {
#   "thinking": "The Results section (node_0013) directly addresses findings...",
#   "node_list": ["node_0013", "node_0029"]
# }
```

### `fetch_node_content(doc_id, node_ids)`
Retrieves the full section text, title, and page number for each selected node from the PageIndex API.

```python
nodes = fetch_node_content(doc_id, ["node_0013", "node_0029"])
# Returns list of dicts with: node_id, title, page, content
```

### `generate_answer(query, node_contents)`
Generates a grounded, cited answer using only the retrieved section content. Explicitly refuses to answer if context is insufficient — no hallucination.

```python
answer = generate_answer("What are the key findings?", nodes)
# Output includes section titles and page number citations
```

### `vectorless_rag(query, tree, doc_id)`
Full end-to-end pipeline combining all three steps.

```python
answer = vectorless_rag(
    query="Summarize the methodology section.",
    tree=pageindex_tree,
    doc_id=doc_id
)
print(answer)
```

### `llm_tree_search_with_expert(query, tree, expert_rules)`
Enhanced retrieval with domain-specific rules injected into the LLM prompt. Guides the model to prioritize certain section types — no fine-tuning needed.

```python
RULES = """
- For methodology questions: prioritize 'Method', 'Approach', 'Framework' sections
- For results questions: prioritize 'Results', 'Findings', 'Analysis' sections
- For recommendations: prioritize 'Conclusion', 'Recommendations' sections
"""

result = llm_tree_search_with_expert(query, tree, RULES)
```

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/Shobhitcld70/Vectorless-RAG.git
cd Vectorless-RAG
```

### 2. Install
```bash
pip install pageindex openai python-dotenv
```

### 3. Set up `.env`
```env
PAGEINDEX_API_KEY=your_pageindex_key
OPENAI_API_KEY=your_openai_key
```

Get your PageIndex API key: [dash.pageindex.ai/api-keys](https://dash.pageindex.ai/api-keys)

### 4. Run
```bash
jupyter notebook Vectorless_RAG_PageIndex.ipynb
```

Update `PDF_PATH` in Cell 4 to point to your PDF and run all cells.

---

## 🔌 Two Modes

### ☁️ Cloud Mode (Default)
Uses PageIndex cloud API. Documents are indexed and stored remotely.
- Fastest setup — just an API key
- Tree index is cached and reusable across sessions
- PageIndex Chat API available (zero LLM setup)

### 🏠 Self-Hosted Mode (Cells 20–23)
Runs entirely locally using the open-source [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) repo.
- Full data privacy — nothing leaves your machine
- Uses your OpenAI key to build the tree locally
- Outputs a `document_pageindex.json` file you control
- Identical RAG pipeline works on local trees

```bash
python run_pageindex.py \
    --pdf_path your_document.pdf \
    --model gpt-4o-2024-11-20 \
    --toc-check-pages 20 \
    --max-pages-per-node 10 \
    --if-add-node-summary yes
```

---

## 💬 Multi-Turn Chat API

Built-in conversational interface with persistent history — no OpenAI setup needed for this mode:

```python
chat_with_document("Give me an overview of this document.", doc_id)
# → "This document covers..."

chat_with_document("What does it say about the methodology?", doc_id)
# → Follows up with context from previous turn
```

---

## 🆚 When to Use Vectorless RAG vs Vector RAG

| Use Vectorless RAG when... | Use Vector RAG when... |
|---|---|
| Documents are long and structured (reports, manuals, legal, research papers) | Documents are short and varied (FAQs, product descriptions) |
| You need traceable, cited answers | Semantic paraphrase matching matters more than structure |
| Domain expertise should guide retrieval | Sub-second retrieval on millions of documents is needed |
| You want to avoid vector DB infrastructure | You're already running a vector store |
| Accuracy is more important than retrieval speed | Speed is the primary constraint |

---

## 📦 Requirements

```
pageindex
openai
python-dotenv
```

---

## 📁 Repository Structure

```
Vectorless-RAG/
├── Vectorless_RAG_PageIndex.ipynb   # Main notebook — 24 cells
├── .env                             # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## 🙋 Author

**Shobhit Krishnan**
- 📧 krishnanshobhit@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/shobhit-krishnan)
- 💻 [GitHub](https://github.com/Shobhitcld70)

---

⭐ If this helped you, consider starring the repo!
