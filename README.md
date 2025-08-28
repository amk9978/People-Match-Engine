# 🎯 Match Engine

**Enterprise-grade Professional Network Analysis Platform**

A sophisticated graph-based matching system that discovers optimal professional communities using advanced multi-feature similarity and AI-powered complementarity analysis. Built for scalable network intelligence with real-time processing capabilities.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=Redis&logoColor=white)](https://redis.io/)
[![NetworkX](https://img.shields.io/badge/NetworkX-013243?style=for-the-badge)](https://networkx.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white)](https://openai.com/)


![1.png](docs/1.png)

![2.png](docs/2.png)


## 🚀 What Makes It Powerful

**Match Engine** revolutionizes professional network analysis by combining traditional graph algorithms with modern AI to find the most strategically valuable professional communities. Unlike simple similarity matching, our system understands the **complementary value** of professional relationships.

### 🎪 **The Magic Behind the System**
1. **🔍 Intelligent Data Processing** - Uploads professional CSV data with automatic deduplication and tag extraction
2. **🤖 AI-Powered Complementarity Analysis** - ChatGPT analyzes complete professional profiles to score strategic relationship value
3. **⚖️ Dynamic Weight Optimization** - Auto-tunes similarity vs complementarity weights based on user intent prompts
4. **🕸️ Advanced Graph Construction** - Builds weighted networks combining embedding similarity with AI-scored complementarity
5. **💎 Dense Subgraph Discovery** - Employs sophisticated algorithms to find the most connected professional communities
6. **📊 Real-Time Insights** - WebSocket-powered live updates with comprehensive visualization and analytics

## 🏗️ System Architecture

### **High-Level System Flow**

## 🛠️ Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- Redis instance

### Installation
```bash
git clone https://github.com/amk9978/people_match_engine
cd match_engine

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key and Redis config

# Launch the system
docker compose up -d
```

### Usage Example
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-User-ID: startup_founder" \
  -F "file=@team_candidates.csv" \
  -F "prompt=I'm building a fintech startup and need complementary technical and business expertise"
```

## 🎨 Core Capabilities

### **🧠 AI-Powered Analysis**
- **Multi-Dimensional Matching** - Analyzes role, experience, persona, industry, market, and offerings
- **Strategic Complementarity Scoring** - ChatGPT evaluates professional synergy potential
- **Intent-Aware Optimization** - Dynamically adjusts matching criteria based on user goals
- **Advanced Graph Algorithms** - Employs density-based subgraph mining and community detection

### **⚡ Performance & Scale**
- **Real-Time Processing** - WebSocket-powered live analysis updates
- **Enterprise Caching** - Redis-backed performance optimization
- **FAISS Integration** - Vector similarity search for large datasets
- **Async Architecture** - Concurrent processing for maximum throughput

### **📊 Professional Analytics**
- **Maximum Weight Cycles** - Discovers optimal professional collaboration chains
- **Community Detection** - Identifies natural professional clusters using Louvain/Greedy Modularity
- **Feature Importance Analysis** - Reveals which attributes drive the strongest connections
- **Interactive Visualizations** - D3.js-powered network graphs with MDS layout

### **🔧 Enterprise Features**
- **Job Persistence** - Redis-backed result storage with job ID retrieval
- **Dataset Versioning** - Track changes, revert, and analyze different data versions
- **Multi-User Support** - User profiles with usage statistics and file management
- **Flexible Data Input** - CSV upload with automatic validation and preprocessing

## 📚 API Documentation

Access the **interactive OpenAPI documentation** at `http://localhost:8000/docs`

### **🔑 Core Endpoints**

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/analyze` | POST | Upload CSV and initiate analysis with optional user prompt |
| `/jobs/{job_id}` | GET | Monitor analysis progress and retrieve status |
| `/jobs/{job_id}/result` | GET | Retrieve completed analysis results from Redis |
| `/users/me` | GET | User profile, statistics, and file management |
| `/datasets/{filename}/add-rows` | POST | Dynamically modify datasets with new entries |
| `/cache/info` | GET | Redis performance metrics and cache statistics |
| `/ws/{client_id}` | WebSocket | Real-time analysis updates and progress monitoring |

### **📤 Response Examples**

<details>
<summary><b>Analysis Results Structure</b></summary>

```json
{
  "job_id": "uuid-string",
  "subgraph_info": {
    "nodes": ["member1", "member2", "member3"],
    "density": 0.85,
    "communities": {
      "community_1": ["member1", "member2"],
      "community_2": ["member3", "member4"]
    },
    "maximum_cycle": {
      "cycle": ["member1", "member2", "member3", "member1"],
      "weight": 0.92,
      "summary": "High-synergy collaboration chain"
    },
    "feature_analysis": {
      "most_important_features": ["experience", "industry"],
      "tuned_weights": {"similarity": 0.6, "complementarity": 0.4}
    },
    "dataset_values": {
      "member1": {"role": "Engineer", "experience": "5 years"},
      "member2": {"role": "Designer", "experience": "3 years"}
    }
  },
  "visualization": {
    "stress_layout": {"coordinates": {...}},
    "edge_weights": {...}
  }
}
```
</details>

## 🚀 Deployment & Scaling

### **Production Configuration**
```bash
# Environment variables
OPENAI_API_KEY=your_api_key
REDIS_URL=redis://localhost:6379
MIN_DENSITY=0.3
MAX_CONCURRENT_JOBS=10
CACHE_TTL=3600

# Docker Compose production setup
docker compose -f docker-compose.prod.yml up -d
```

### **Performance Optimization**
- **FAISS Integration** - Handles datasets with 10K+ professionals
- **Redis Clustering** - Distributed caching for horizontal scale
- **Async Processing** - Concurrent matrix building and API operations
- **Smart Caching** - Embedding and graph result persistence

## 🔮 Future Roadmap

### **🎯 Phase 1: Enhanced Analytics**
- Advanced subgraph algorithms (k-core decomposition, modularity optimization)
- ML-based complementarity scoring with custom training
- Multi-objective optimization for complex matching criteria

### **🎯 Phase 2: Enterprise Integration**
- Multi-tenant architecture with organization isolation
- API rate limiting and usage analytics
- Export capabilities (PDF reports, Gephi files, network visualizations)

### **🎯 Phase 3: Advanced Features**
- Real-time collaboration scoring updates
- Integration with LinkedIn, GitHub, and professional APIs
- Predictive analytics for team formation success

---
