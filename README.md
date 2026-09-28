![AI/ML Banner](linkedin%20banner.png)

# Hi there 👋 I'm Sahil Sawant

**AI Engineer | Agentic AI & RAG | LLM Applications** · Minneapolis, MN

---

### 🚀 About
AI engineer with 2 years of experience building production LLM systems: multi-agent workflows, hybrid RAG over knowledge graphs and vector stores, and the backend services around them. I like turning messy, real-world data and ambiguous requests into systems people can trust, which usually means showing the evidence behind an answer and measuring how well it works.

MS in Artificial Intelligence from the University at Buffalo. Currently doing research on MyVictor at UB's A2IL lab, and open to AI Engineer and Forward Deployed Engineer roles.

### 💼 Experience

**Volunteer Research Scientist | A2IL Lab, University at Buffalo** (Jun 2026 – Present)
* Built **MyVictor**, a RAG chatbot over 450+ UB CSE department pages, combining a Neo4j knowledge graph with a Qdrant vector store (BGE-M3 dense + BM25 hybrid search) in a single retrieval layer that runs both paths in parallel.
* Cut median response latency from ~90s to ~45s by disabling the model's internal reasoning on Cypher generation while keeping it on for answer synthesis.
* Scraped and normalized HTML, Markdown, and JSONL from Playwright/Firecrawl crawls into a graph of faculty, courses, labs, research areas, and policies.

**AI Software Engineer | Arta Support** (Mar 2026 – May 2026)
* Orchestrated a 4-agent troubleshooting workflow (router, intent, clarification, diagnosis) on Claude Sonnet with ~30s response latency, using prompt caching, Socket.IO tool-calling, and conditional self-correction retries.
* Paired Pinecone with Neo4j in a graph-guided RAG system, improving retrieval precision by 30%.
* Automated a PDF-to-knowledge-graph ingestion pipeline, cutting knowledge-base update time by 40%.
* Built AWS authentication and LLM token accounting (Cognito, DynamoDB, KMS, EC2): role-based dashboards, self-serve API key rotation, and per-model usage tracking for budget controls.

**Analyst | Capgemini** (Dec 2022 – May 2024)
* Built a text-to-SQL RAG system (LangGraph, Llama-2-70B) over Oracle EBS schemas so non-technical warehouse staff could query aerospace inventory in plain language, handling 5,000+ weekly queries with a 94% user thumbs-up rate.
* Cut prompt tokens by 65% and eliminated hallucinated column names by retrieving live schema metadata per question instead of using a static prompt.
* Supported production Oracle EBS batch workflows: resolved 100+ incidents with 100% SLA compliance and optimized SQL/PL-SQL for a 40% faster run time.

> Arta and Capgemini work is confidential, so there's no public code for it.

### 🛠️ Skills
* **GenAI & Agents:** Agentic workflows (LangGraph, LangChain, pydantic-ai), MCP, RAG and RAG evaluation (RAGAS), Cohere reranking, LLM orchestration, prompt engineering, Claude, OpenAI, Gemini, Hugging Face
* **Languages:** Python, SQL/PL-SQL, PySpark, JavaScript
* **Backend & Frontend:** FastAPI, Node.js, React, Socket.IO, REST APIs, Git, Linux
* **Data & Cloud:** Neo4j (Cypher), Qdrant, Pinecone, DuckDB, AWS (Bedrock, Cognito, DynamoDB, KMS, EC2, SageMaker), Databricks, Oracle EBS
* **ML & MLOps:** PyTorch, TensorFlow, Scikit-Learn, CNNs/Transformers, Docker, GitHub Actions CI/CD, model monitoring and versioning, Streamlit
* **Computer Vision & Robotics:** YOLOv8, OpenCV, ONNX, ROS2, Unitree Go2

### 📁 Key Projects
* [**B.O.L.T.**](https://github.com/sahillarious/BOLT): Vision-guided object following on a Unitree Go2 quadruped. Quantized YOLOv8n via ONNX on a Jetson Nano (mAP@0.5 of 0.995), with perception decoupled from a finite-state motion controller. 48 ms average end-to-end latency, >90% tracking success indoors.
* [**DeepSpeech Therapy**](https://github.com/sahillarious/DeepSpeech): Real-time voice coach. A 1D convolutional autoencoder detects speech clarity issues (96.89% accuracy), a CNN + Transformer diagnoses stutter types, and a GPT-4o agent gives context-aware articulation feedback.
* [**DermAI**](https://github.com/sahillarious/DermAI): Multimodal dermatology assistant. A vision ensemble (ResNet50, DenseNet121, VGG-19, EfficientNet-B0) is exposed as a classification tool to a Gemini 3 Flash LLM for feedback across 7 lesion categories on HAM10000. SMOTE and class-weighted training raised precision from 87% to 94%.
* [**Buffalo Accident Risk Prediction & Resource Allocation**](https://github.com/sahillarious/Buffalo-Accident-Risk-Prediction-Resource-Allocation): Multi-agent reinforcement learning (PPO) system that reallocates 145 emergency response units from live data feeds.
* **End-to-End ML Pipeline for Job-Market Classification:** Five-class classifier at 87% accuracy on 1.3M+ LinkedIn postings (PySpark on Databricks), served on a SageMaker endpoint with Model Monitor drift detection and GitHub Actions retrain-and-redeploy.

### 🏆 Achievements
* **2nd Place:** University at Buffalo AI for Good Challenge (Spring 2025), an LSTM-based snow management solution
* **Best Student Volunteer**, Standing Committee of the Section – SAC (IEEE Bombay Section, 2022)
* **Winner:** IEEE R10 Connect Logo Design Competition

### 🎓 Education
* **MS in Artificial Intelligence**, University at Buffalo (Dec 2025)
* **BE in Electronics & Telecommunication**, University of Mumbai (May 2022)

### 📬 Contact & Connect
* **LinkedIn:** [linkedin.com/in/sahilsawant01](https://linkedin.com/in/sahilsawant01)
* **Email:** [sahilshivajisawant@gmail.com](mailto:sahilshivajisawant@gmail.com)
* **Portfolio:** [sahillarious.github.io](https://sahillarious.github.io)
