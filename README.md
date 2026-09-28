![AI/ML Banner](linkedin%20banner.png)

# Hi there 👋 I'm Sahil Sawant

**AI Engineer | Agentic AI & RAG | Document Intelligence** · Minneapolis, MN

---

### 🚀 About
AI engineer with 2 years of experience building production LLM systems: multi-agent workflows, hybrid RAG over knowledge graphs and vector stores, and document-processing pipelines that turn unstructured content (PDFs, scanned images, web pages) into structured knowledge for AI assistants. I like building systems people can trust, which usually means showing the evidence behind an answer and measuring how well it works.

MS in Artificial Intelligence from the University at Buffalo. Currently doing research on MyVictor at UB's A2IL lab, and open to AI Engineer and Forward Deployed Engineer roles.

### 💼 Experience

**Volunteer Research Scientist | A2IL Lab, University at Buffalo** (Jun 2026 – Present)
* Built **MyVictor**, a RAG chatbot over 450+ UB CSE department and publication pages. The ingestion pipeline turns scraped web content (HTML, Markdown, JSONL from Playwright/Firecrawl crawls) into a Neo4j knowledge graph of faculty, courses, labs, research areas, and policies, alongside a Qdrant hybrid store (BGE-M3 dense + BM25).
* Runs graph and vector retrieval in parallel as a single retrieval layer, so a Qwen-27B model can answer from both structured relationships and descriptive source text.
* Classified publications and faculty content by research topic, metadata, and canonical concepts.
* Built a resume-to-faculty matching feature: one LLM extraction pass pulls research concepts from an uploaded resume, then a deterministic scoring pipeline ranks 105 faculty profiles with exact-match citations, so results are reproducible and verifiable.
* Cut response latency to ~45s by routing ~60% of queries through deterministic regex-based logic, reserving LLM-generated Cypher for edge cases, and disabling the model's internal reasoning on Cypher generation while keeping it on for answer synthesis.

**AI Software Engineer | Arta Support** (Mar 2026 – May 2026)
* Orchestrated a 4-agent troubleshooting workflow (router, intent, clarification, diagnosis) powered by Claude Sonnet, with ~30s end-to-end latency, using prompt caching, Socket.IO tool-calling, and conditional self-correction retries.
* Automated an IDP pipeline for PDF runbooks and incident documents: OCR on embedded images, extraction and classification of text and tables, and conversion into structured knowledge for an AI copilot.
* Built a PDF-to-knowledge-graph ingestion pipeline (entity and relationship extraction, generated Cypher into Neo4j) that cut knowledge-base update time by 40% and kept vector indices in sync with source content.
* Paired Pinecone with Neo4j in a graph-guided RAG system, improving retrieval precision by 30% by landing agents on exact graph nodes instead of stuffing context.
* Implemented role-based access control, API-key lifecycle policies (expiration, scheduled disablement, revocation, self-serve rotation, access history), and per-model LLM usage tracking for budget controls using AWS Cognito, DynamoDB, KMS, and EC2.

**Analyst | Capgemini** (Dec 2022 – May 2024)
* Built a text-to-SQL RAG system (LangGraph, Llama-2-70B) over Oracle EBS schemas so non-technical warehouse staff could query aerospace inventory in plain language, using a FastAPI backend and React frontend that show the generated SQL, raw rows, and charts. Handled 5,000+ weekly queries with a 94% user thumbs-up rate.
* Cut prompt tokens by 65% and eliminated hallucinated column names by retrieving live schema metadata per question instead of using a static prompt.
* Supported Oracle EBS workflows for an aerospace client (1M+ daily records across Order Management, Procurement, and Inventory Control): resolved 100+ production incidents across AppWorx batch jobs with 100% SLA compliance, and optimized SQL/PL-SQL extraction, validation, and reporting for a 40% faster run time.
* Worked with business and technical stakeholders to gather requirements, validate data, and support production deployments.

> Arta and Capgemini work is confidential, so there's no public code for it.

### 🛠️ Skills
* **GenAI & Agents:** Agentic workflows (LangGraph, LangChain, CrewAI, pydantic-ai), MCP, RAG and RAG evaluation (RAGAS), embeddings, semantic search, Cohere reranking, knowledge graphs, LLM orchestration, prompt engineering, Claude, OpenAI, Gemini, Hugging Face
* **Document Processing:** OCR, document ingestion, classification and information extraction, metadata processing, AI-powered retrieval
* **Languages:** Python, SQL/PL-SQL, PySpark, JavaScript
* **Backend & Frontend:** FastAPI, Node.js, React, Socket.IO, REST APIs, System Design, Git, Linux
* **Data & Cloud:** Neo4j (Cypher), Qdrant, Pinecone, FAISS, DuckDB, NoSQL, AWS (Bedrock, Cognito, DynamoDB, KMS, EC2, SageMaker), Databricks, Apache Spark, Oracle EBS
* **ML & MLOps:** PyTorch, TensorFlow, Scikit-Learn, CNNs/Transformers, Docker, GitHub Actions CI/CD, model monitoring and versioning, Streamlit, Gradio
* **Computer Vision & Robotics:** YOLOv8, OpenCV, ONNX, ROS2, Unitree Go2

### 📁 Key Projects
* [**B.O.L.T.**](https://github.com/sahillarious/BOLT): Vision-guided object following on a Unitree Go2 quadruped. Quantized YOLOv8n via ONNX on a Jetson Nano (mAP@0.5 of 0.995), with perception decoupled from a finite-state motion controller. 48 ms average end-to-end latency, >90% tracking success indoors.
* [**DeepSpeech Therapy**](https://github.com/sahillarious/DeepSpeech): Real-time voice coach. A 1D convolutional autoencoder detects speech clarity issues (96.89% accuracy), a CNN + Transformer diagnoses stutter types, and a GPT-4o agent gives context-aware articulation feedback.
* [**DermAI**](https://github.com/sahillarious/DermAI): Multimodal dermatology assistant. A vision ensemble (ResNet50, DenseNet121, VGG-19, EfficientNet-B0) trained on 10,015 images is exposed as a classification tool to a Gemini 3 Flash LLM for feedback across 7 lesion categories on HAM10000. SMOTE and class-weighted training raised precision from 87% to 94%.
* **Sanskrit OCR:** Dashboard-driven pipeline that segments scanned documents into Devanagari characters with a CNN (93% character-recognition accuracy), translates the output, and stores structured results in S3.
* [**Buffalo Accident Risk Prediction & Resource Allocation**](https://github.com/sahillarious/Buffalo-Accident-Risk-Prediction-Resource-Allocation): Multi-agent reinforcement learning (PPO) system that reallocates 145 emergency response units from live data feeds.
* **End-to-End ML Pipeline for Job-Market Classification:** Five-class classifier at 87% accuracy on 1.3M+ LinkedIn postings (PySpark on Databricks), served on a SageMaker endpoint with Model Monitor drift detection and GitHub Actions retrain-and-redeploy.
* **Jobmail:** Local Python tool that parses 350+ weekly job-application emails with a deterministic regex pipeline, de-duplicates and prioritizes them by role, and drafts resume-matched outreach.

### 🏆 Achievements
* **2nd Place:** University at Buffalo AI for Good Challenge (Spring 2025). LSTM model fed by live weather-API data to recommend snow-shoveling windows, with role-based access for residents, contractors, and municipal users to book snow removal.
* **Best Student Volunteer**, Standing Committee of the Section – SAC (IEEE Bombay Section, 2022)
* **Winner:** IEEE R10 Connect Logo Design Competition

### 🎓 Education
* **MS in Artificial Intelligence**, University at Buffalo (Aug 2024 – Dec 2025)
* **BE in Electronics & Telecommunication**, University of Mumbai (Aug 2018 – May 2022)

### 📬 Contact & Connect
* **LinkedIn:** [linkedin.com/in/sahilsawant01](https://linkedin.com/in/sahilsawant01)
* **Email:** [sahilshivajisawant@gmail.com](mailto:sahilshivajisawant@gmail.com)
* **Portfolio:** [sahillarious.github.io](https://sahillarious.github.io)
