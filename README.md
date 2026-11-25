# Anand Wealth AI Chatbot
### A Retrieval-Augmented Generation (RAG) Chatbot for Financial Services

An intelligent, customer-facing AI chatbot developed for **Anand Wealth Consultants** using Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), hybrid search and semantic reranking.

The chatbot is designed to answer customer queries about financial services, mutual funds, SIPs, retirement planning and other company-related information using a dedicated knowledge base.

Unlike a conventional chatbot that relies entirely on an LLM's pretrained knowledge, this application retrieves relevant information from a knowledge base before generating responses. This helps improve factual consistency and reduces the likelihood of unsupported answers.

---

## 1. Project Overview

The objective of this project is to build a practical AI-powered customer support assistant that can:

- Understand natural-language customer queries.
- Retrieve relevant information from a company knowledge base.
- Answer general financial questions in simple language.
- Maintain conversational context across multiple messages.
- Identify questions that cannot be answered using available information.
- Avoid generating unsupported company-specific information.
- Maintain a consistent and professional company voice.
- Provide appropriate fallback responses when information is unavailable.

The chatbot uses a local LLM through Ollama, allowing the application to run without depending on paid external LLM APIs.

### Key Highlights

- Retrieval-Augmented Generation (RAG).
- Dense semantic retrieval using vector embeddings.
- Sparse retrieval using BM25.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Cross-Encoder-based reranking.
- Contextual question rewriting.
- Three-level answerability classification.
- Small-talk and intent detection.
- Financial safety instructions.
- Streamlit-based conversational interface.
- Local LLM inference using Ollama.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Runtime | Ollama |
| Primary LLM | Qwen2.5:7b / Llama 3.2:3b (evaluation) |
| Embedding Model | nomic-embed-text |
| Sparse Retrieval | BM25 |
| Dense Retrieval | Cosine Similarity |
| Hybrid Search | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Reranker Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Numerical Processing | NumPy |
| Model Integration | Ollama Python Library |

*Note: Multiple LLMs are being evaluated to identify the best balance between response quality, instruction following and local inference performance. The final production model is yet to be established.*

---

## 3. System Architecture

The application follows a multi-stage RAG pipeline.

### Architecture Flow

```text
                  Customer Message
                         |
                         v
                   Intent Detection
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Small Talk                RAG Query
             |                       |
             v                       v
      Direct Response       Contextual Question
                                Rewriting
                                    |
                                    v
                          Hybrid Retrieval
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
                   Dense Search           BM25 Search
                         |                     |
                         +----------+----------+
                                    |
                                    v
                              RRF Fusion
                                    |
                                    v
                         Cross-Encoder Reranking
                                    |
                                    v
                           Relevance Filtering
                                    |
                                    v
                        Answerability Classification
                                    |
                 +------------------+------------------+
                 |                  |                  |
                 v                  v                  v
          FULLY_ANSWERABLE  PARTIALLY_ANSWERABLE  NOT_ANSWERABLE
                 |                  |                  |
                 v                  v                  v
          Generate Answer    Generate Limited      Fallback
                              Answer                Response
                 |                  |                  |
                 +------------------+------------------+
                                    |
                                    v
                           Response Formatting
                                    |
                                    v
                              Final Answer
```

### Explanation of the Pipeline

**1. Intent Detection**

The chatbot first identifies the type of incoming message.

It distinguishes between:

- Greetings.
- Thank-you messages.
- Questions about the chatbot.
- Actual knowledge-based queries.

Simple conversational messages receive direct responses without unnecessarily executing the complete RAG pipeline.

**2. Contextual Question Rewriting**

Follow-up questions often contain references such as "it", "they", "that" or "its".

For example:

- First question: What is a SIP?
- Follow-up: What are its benefits?

The rewriting component uses conversation history to transform the follow-up into a standalone question:

"What are the benefits of a SIP?"

This allows the retrieval system to search for the correct information.

**3. Dense Retrieval**

The embedding model, `nomic-embed-text`, converts text into numerical vector representations.

The application generates embeddings for the knowledge-base chunks and compares them with the question embedding using cosine similarity.

This helps retrieve semantically related information, even when the wording of the question differs from the knowledge base.

**4. Sparse Retrieval using BM25**

BM25 provides keyword-based retrieval.

It identifies documents based on matching terms between the question and the knowledge base.

This complements dense retrieval by capturing important keywords, financial terminology and exact phrases.

**5. Hybrid Retrieval using Reciprocal Rank Fusion**

Dense retrieval and BM25 have different strengths.

Instead of relying on either method independently, the application combines their rankings using Reciprocal Rank Fusion.

RRF calculates a combined ranking score based on the positions of documents in the individual retrieval results.

This helps balance semantic similarity and keyword matching.

**6. Cross-Encoder Reranking**

The retrieved candidates are passed through a Cross-Encoder model.

Unlike embedding-based retrieval, which compares independently generated vectors, the Cross-Encoder evaluates the relationship between the query and each retrieved passage together.

It assigns relevance scores and reorders the candidates to prioritise more relevant information.

**7. Answerability Classification**

The chatbot uses three answerability categories:

| Category | Meaning |
|---|---|
| FULLY_ANSWERABLE | The retrieved information is sufficient to answer the question. |
| PARTIALLY_ANSWERABLE | Some parts can be answered, but other information is missing. |
| NOT_ANSWERABLE | The available information does not sufficiently support an answer. |

This stage is important because retrieving a relevant passage does not necessarily mean that the passage contains the actual answer.

**8. Response Generation**

The LLM receives the customer's question, retrieved context and system instructions.

It generates a response while following the company's communication guidelines.

The system prompt also establishes financial safety restrictions, including avoiding guaranteed-return claims and personalised investment recommendations.

**9. Response Formatting**

The final response is formatted to maintain a consistent first-person company voice.

For example, responses should use "we", "our" and "us" instead of unnecessarily referring to Anand Wealth Consultants as an external organisation.

---

## 4. Major Features

### Hybrid Retrieval

Combines semantic and keyword-based retrieval to improve the quality of retrieved information.

### Context-Aware Conversations

Uses previous conversation messages to understand follow-up questions.

### Intelligent Answerability Detection

Distinguishes between fully answerable, partially answerable and unanswerable questions.

### Hallucination Prevention

Uses retrieved context, answerability classification and system instructions to reduce unsupported responses.

### Financial Safety

The chatbot follows restrictions designed for financial services:

- Does not guarantee investment returns.
- Does not describe investments as risk-free.
- Does not provide personalised investment recommendations.
- Does not invent company fees, policies or business information.
- Directs customers to the company's team when confirmed information is unavailable.

### Local LLM Inference

Uses Ollama to run language models locally instead of depending on paid external LLM APIs.

### Conversational Interface

Streamlit provides a chat interface with conversation history and customer-facing responses.

---

## 5. Challenges Faced During Development

Building this application involved more than simply connecting an LLM to a document. Several challenges arose while developing the retrieval, reasoning and response-generation pipeline.

### Challenge 1: Retrieving Relevant Information

A basic vector search system does not always retrieve the most appropriate passage.

Some questions depend on exact financial terminology, while others express the same concept using different wording.

**Solution:**

Implemented hybrid retrieval using dense semantic search and BM25, followed by Reciprocal Rank Fusion.

This combines the strengths of semantic and keyword-based retrieval.

### Challenge 2: Improving Retrieval Precision

Even when the initial retrieval stage returns relevant passages, the ranking may not always be optimal.

**Solution:**

Integrated a Cross-Encoder reranker to evaluate query-passage relevance and reorder retrieved candidates.

### Challenge 3: Understanding Follow-up Questions

Questions such as "What are its benefits?" are difficult to process independently because their meaning depends on previous messages.

**Solution:**

Implemented contextual question rewriting using conversation history.

This converts ambiguous follow-up questions into standalone queries before retrieval.

### Challenge 4: Preventing Hallucinations

An LLM can generate convincing answers even when the knowledge base does not contain the required information.

This is particularly problematic in financial applications.

**Solution:**

Introduced answerability classification with three possible outcomes and fallback responses for unsupported questions.

### Challenge 5: Distinguishing Relevance from Answerability

One important observation during development was that a high Cross-Encoder score does not necessarily mean that the retrieved information contains the answer.

For example, a passage discussing SIPs may be highly relevant to a question about guaranteed returns without actually confirming the company's policy.

**Solution and ongoing improvement:**

The answerability classification stage is being refined so that retrieval relevance is not treated as proof of answerability.

Automatic answerability overrides based solely on Cross-Encoder scores are being removed in favour of more reliable evidence-based decisions.

### Challenge 6: Maintaining a Consistent Company Voice

LLMs sometimes respond using third-person descriptions of the company, such as "Anand Wealth Consultants provides..."

However, a customer-facing company chatbot should communicate as a member of the organisation.

**Solution:**

Added first-person response instructions and response-formatting logic to maintain consistent company language.

### Challenge 7: Local Model Performance

Running language models locally introduces hardware limitations, particularly when using larger models.

The available GPU memory, context length and model size influence inference performance.

**Solution and ongoing improvement:**

Using Ollama for local inference and evaluating different model sizes to identify an appropriate balance between response quality and resource consumption.

---

## 6. Model Evaluation

Multiple local language models are being considered for the generation stage.

Current candidates include:

- Llama 3.2:3b
- Llama 3.1:8b
- Qwen2.5:7b

The objective is to evaluate their performance using an identical set of questions and an unchanged RAG pipeline.

### Evaluation Criteria

| Metric | Evaluation |
|---|---|
| Factual Accuracy | Whether the generated response contains correct information. |
| Retrieval Relevance | Whether the retrieved passages support the question. |
| Instruction Following | Whether the model follows the system prompt. |
| Hallucination Prevention | Whether unsupported information is avoided. |
| Response Quality | Clarity, readability and professional tone. |
| Response Speed | Time required to generate the response. |

A dedicated evaluation dataset is being developed to test:

- Basic company-related questions.
- General financial questions.
- Context-dependent follow-up questions.
- Questions with missing information.
- Personalised investment requests.
- Guaranteed-return questions.
- Attempts to override financial safety instructions.

Formal benchmark results have not yet been established.

---

## 7. Current Limitations

Although the application has several advanced RAG components, there are areas that require further work before production deployment.

1. **Knowledge Base Reliability**

   The current knowledge base is based on publicly available company website information. It requires verification and approval before being treated as an authoritative source.

2. **Answerability Accuracy**

   The three-category classifier needs systematic evaluation using labelled questions and retrieved passages.

3. **Retrieval Evaluation**

   There is currently no comprehensive quantitative evaluation of retrieval quality using metrics such as Recall@K, Precision@K and MRR.

4. **Question Rewriting**

   Contextual rewriting can occasionally introduce unnecessary information from previous messages. It needs stronger constraints to preserve the original question's intent.

5. **Error Handling**

   Failures involving Ollama, embedding generation, reranking and LLM inference need consistent handling so that one failed component does not unexpectedly interrupt the application.

6. **Conversation Management**

   Conversation history needs better management, including context limits and long-conversation handling.

7. **Security and Privacy**

   The application requires additional security, privacy and access-control considerations before being used with real customer information.

8. **Production Deployment**

   The current implementation is a local prototype. Deployment requires further work involving hosting, monitoring, reliability and operational security.

---

## 8. Future Improvements

The following improvements are planned to make the application more reliable and suitable for real-world use.

### Retrieval Improvements

- Evaluate different chunk sizes and overlap configurations.
- Implement a labelled retrieval evaluation dataset.
- Measure Recall@K, Precision@K, MRR and nDCG.
- Improve retrieval filtering and relevance decisions.
- Experiment with different embedding models.

### Answer Generation Improvements

- Compare Qwen2.5:7b and Llama3.1:8b using a fixed evaluation dataset.
- Improve answerability classification.
- Introduce more structured response validation.
- Reduce unsupported claims through evidence-based generation.
- Improve contextual question rewriting.

### Reliability Improvements

- Add robust error handling for Ollama and model failures.
- Implement fallback retrieval when dense embedding generation fails.
- Add structured logging.
- Introduce automated testing for individual pipeline components.
- Monitor response latency and resource consumption.

### Financial Safety Improvements

- Strengthen protection against prompt injection.
- Improve validation of generated financial responses.
- Ensure that company-specific claims are supported by approved sources.
- Improve handling of personalised financial questions.
- Ensure that unsupported financial claims are rejected or appropriately qualified.

### Production Improvements

- Deploy the application using a suitable hosting environment.
- Introduce a persistent conversation database if required.
- Add authentication and access controls where appropriate.
- Implement monitoring and error tracking.
- Establish a process for regularly updating and validating the knowledge base.

---

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Ollama installed.
- Git.
- Sufficient system memory and compute resources for the selected language model.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd anand-wealth-rag-chatbot
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Ollama Models

Install the embedding model:

```bash
ollama pull nomic-embed-text
```

Download a language model:

```bash
ollama pull qwen2.5:7b
```

Alternatively, use:

```bash
ollama pull llama3.2:3b
```

The selected model must match the configuration in the application.

### Step 5: Run Ollama

Ensure that the Ollama service is running.

### Step 6: Start the Application

```bash
streamlit run test_playright.py
```

The application should open in your default browser.

---

## 10. Project Structure

```text
anand-wealth-rag-chatbot/
│
├── data/
│   └── anand_wealth_website_knowledge_base.txt
│
├── test_playright.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

The project structure may evolve as the application is refactored into separate modules for retrieval, reranking, answerability and response generation.

---

## 11. Learning Outcomes

This project provided practical experience with several important concepts in modern Generative AI development.

Key learning areas include:

- Understanding the complete Retrieval-Augmented Generation pipeline.
- Working with local LLMs through Ollama.
- Generating and comparing text embeddings.
- Implementing dense and sparse information retrieval.
- Combining retrieval methods using Reciprocal Rank Fusion.
- Understanding Cross-Encoder reranking.
- Managing contextual conversations.
- Designing answerability classification.
- Writing system prompts for controlled LLM behaviour.
- Identifying hallucination and reliability problems.
- Evaluating language models for a specific application.
- Building a conversational application using Streamlit.

---

## 12. Conclusion

The Anand Wealth AI Chatbot is an ongoing project focused on building a practical, context-aware and reliable RAG-based customer support assistant for financial services.

The project goes beyond basic document retrieval by combining hybrid search, reranking, contextual question rewriting and answerability classification into a multi-stage pipeline.

An important focus throughout development has been understanding the limitations of LLMs and designing mechanisms to reduce unsupported responses, especially in a financial services environment.

The next development phase focuses on systematic evaluation, improved reliability, stronger financial safety and preparation for deployment.

**Project Status:** Active Development

---

### Disclaimer

This project is a technical AI application prototype. Its responses should not be considered financial, investment, tax or legal advice. Company-specific information must be verified against approved company sources before customer-facing deployment.
