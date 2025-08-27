# Match Engine
A professional network matching system that finds optimal dense subgraphs of people using multi-feature similarity and complementarity analysis. Takes optional user prompts to tailor matching toward specific intents (hiring, networking, partnerships).

## How It Works
1. **Upload CSV** with professional data (roles, experience, industry, etc.)
2. **AI Analysis** builds complementarity matrices using ChatGPT for strategic relationship scoring
3. **Graph Construction** creates weighted edges combining similarity + complementarity scores  
4. **Dense Subgraph Mining** finds the most connected group using iterative peeling algorithm
5. **Real-time Updates** via WebSocket for job progress

## Prerequisites
1. Install Docker and Docker Compose
2. Clone the repository
3. Fill the .env accordingly (OpenAI API key, Redis config)
4. `docker compose up -d`

## Example
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-User-ID: user123" \
  -F "file=@professionals.csv" \
  -F "prompt=I want to hire for my startup"
```

## Architecture
**FastAPI** async backend with **Redis** caching, **NetworkX** graph algorithms, and **ChatGPT** complementarity analysis. Uses `asyncio` for concurrent matrix building and API operations.

**Key Components:**
- `MatrixBuilder` - Builds complementarity matrices using complete profile vectors (async)
- `GraphBuilder` - Creates weighted graphs and finds dense subgraphs
- `SimilarityCalculator` - Computes embedding-based feature similarities
- `EmbeddingBuilder` - Handles tag deduplication and embeddings (async)

## Features
- **Multi-feature matching**: Role, experience, persona, industry, market, offering
- **Complementarity scoring**: ChatGPT analyzes strategic value of professional connections
- **User intent tuning**: Adjusts similarity/complementarity weights based on prompts
- **Dataset versioning**: Track changes, revert, and analyze different versions
- **Real-time processing**: WebSocket updates for long-running analyses

### API Documentation
Visit `/docs` for interactive OpenAPI documentation.

**Core Endpoints:**
- `POST /analyze` - Upload CSV and start analysis
- `GET /jobs/{job_id}` - Check analysis status
- `GET /users/me` - User profile and statistics
- `POST /datasets/{filename}/add-rows` - Modify datasets
- `GET /cache/info` - Redis cache statistics

## Next Steps
- FAISS optimization for large datasets
- Advanced subgraph algorithms (k-core, modularity)
- ML-based complementarity scoring
- Multi-tenant isolation
- Export formats (PDF reports, network visualizations)