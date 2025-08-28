#!/usr/bin/env python3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from infrastructure.container import container

# Import existing presentation models that are still needed
from presentation.models import AnalysisResponse, JobStatus

# Import the old services for backward compatibility
from services.analysis_service import analysis_service, file_service
from services.cache_service import cache_service
from services.notification_service import notification_service

app = FastAPI(
    title="Graph Matcher API",
    description="Dense subgraph analysis with multi-feature embedding matching - Refactored Architecture",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers from the container
for router in container.get_all_routers():
    app.include_router(router)


# Health check endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Graph Matcher API v2.0 with Clean Architecture",
        "status": "healthy",
        "architecture": "Domain-Driven Design with Clean Architecture",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    cache_health = cache_service.health_check()
    if cache_health["status"] != "healthy":
        return {
            "status": "degraded",
            "redis": cache_health["redis"],
            "message": "Cache service unhealthy",
        }

    return {
        "status": "healthy",
        "redis": cache_health["redis"],
        "architecture": "Layered Architecture",
        "layers": ["Domain", "Infrastructure", "Application", "Presentation"],
    }


# Legacy endpoints for backward compatibility can be added here
# but ideally they should be migrated to the new architecture

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
