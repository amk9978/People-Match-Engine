#!/usr/bin/env python3

import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from graph_matcher import GraphMatcher
from hyperparameter_tuner import HyperparameterTuner
from preprocess_embeddings import EmbeddingPreprocessor
from semantic_person_deduplicator import SemanticPersonDeduplicator


class AnalysisService:
    """Service for handling graph analysis operations"""

    def __init__(self):
        self.jobs = {}

    def _serialize_numpy(self, obj):
        """Convert numpy objects to JSON serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy(item) for item in obj]
        return obj

    def create_job(
        self,
        filename: str,
        min_density: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Create a new analysis job"""
        job_id = str(uuid.uuid4())

        self.jobs[job_id] = {
            "status": "queued",
            "progress": "Job created",
            "result": None,
            "error": None,
            "timestamp": datetime.now(),
            "filename": filename,
            "min_density": min_density,
            "prompt": prompt,
        }

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> Dict:
        """Get all jobs summary"""
        return {
            "jobs": [
                {
                    "job_id": job_id,
                    "status": data["status"],
                    "timestamp": data["timestamp"],
                    "filename": data.get("filename", "unknown"),
                }
                for job_id, data in self.jobs.items()
            ]
        }

    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: str = None,
        result: Dict = None,
        error: str = None,
    ):
        """Update job status"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            if progress:
                self.jobs[job_id]["progress"] = progress
            if result:
                self.jobs[job_id]["result"] = result
            if error:
                self.jobs[job_id]["error"] = error
            self.jobs[job_id]["timestamp"] = datetime.now()

    async def run_analysis(
        self,
        job_id: str,
        csv_path: str,
        notification_service,
        min_density: float = None,
        prompt: Optional[str] = None,
    ):
        """Run graph analysis with progress notifications"""
        try:
            self.update_job_status(job_id, "processing", "Initializing GraphMatcher...")
            await notification_service.broadcast_job_update(
                job_id,
                {"status": "processing", "progress": "Initializing GraphMatcher..."},
            )

            matcher = GraphMatcher(csv_path, min_density)

            self.update_job_status(job_id, "processing", "Loading data...")
            await notification_service.broadcast_job_update(
                job_id, {"status": "processing", "progress": "Loading data..."}
            )
            matcher.load_data()

            # STEP 1: Semantic deduplication to create curated tags
            self.update_job_status(
                job_id, "processing", "Running semantic tag deduplication..."
            )
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "processing",
                    "progress": "Running semantic tag deduplication...",
                },
            )

            deduplicator = SemanticPersonDeduplicator()
            await deduplicator.run_semantic_deduplication(csv_path, threshold=0.75)

            # STEP 2: Preprocess embeddings for curated tags only
            self.update_job_status(
                job_id, "processing", "Preprocessing curated tag embeddings..."
            )
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "processing",
                    "progress": "Preprocessing curated tag embeddings...",
                },
            )

            preprocessor = EmbeddingPreprocessor()
            await preprocessor.preprocess_all_embeddings(
                csv_path
            )  # Now uses curated tags

            # STEP 3: Create feature embeddings (now uses cached embeddings)
            self.update_job_status(
                job_id, "processing", "Creating feature embeddings..."
            )
            await notification_service.broadcast_job_update(
                job_id,
                {"status": "processing", "progress": "Creating feature embeddings..."},
            )
            feature_embeddings = await matcher.embed_features()

            # Tune hyperparameters based on user prompt (if provided)
            if prompt:
                self.update_job_status(
                    job_id, "processing", "Tuning hyperparameters for user intent..."
                )
                await notification_service.broadcast_job_update(
                    job_id,
                    {
                        "status": "processing",
                        "progress": "Tuning hyperparameters for user intent...",
                    },
                )

                tuner = HyperparameterTuner()
                tuned_weights = await tuner.tune_and_apply(prompt)

                # Store the tuned weights in job metadata
                self.jobs[job_id]["tuned_weights"] = tuned_weights
                self.jobs[job_id]["user_prompt"] = prompt

            # Choose graph creation method based on settings
            use_faiss = os.getenv("USE_FAISS_OPTIMIZATION", "false").lower() == "true"
            if use_faiss:
                self.update_job_status(
                    job_id, "processing", "Building FAISS-optimized graph..."
                )
                await notification_service.broadcast_job_update(
                    job_id,
                    {
                        "status": "processing",
                        "progress": "Building FAISS-optimized similarity graph...",
                    },
                )
                top_k = int(os.getenv("FAISS_TOP_K", "50"))
                matcher.create_graph_faiss(feature_embeddings, top_k=top_k)
            else:
                self.update_job_status(job_id, "processing", "Building graph...")
                await notification_service.broadcast_job_update(
                    job_id,
                    {
                        "status": "processing",
                        "progress": "Building similarity graph...",
                    },
                )
                matcher.create_graph(feature_embeddings)

            self.update_job_status(job_id, "processing", "Finding dense subgraph...")
            await notification_service.broadcast_job_update(
                job_id,
                {"status": "processing", "progress": "Finding dense subgraph..."},
            )
            largest_dense_nodes, density = matcher.find_largest_dense_subgraph()

            self.update_job_status(job_id, "processing", "Analyzing results...")
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "processing",
                    "progress": "Analyzing results and generating insights...",
                },
            )
            result = matcher.get_subgraph_info(largest_dense_nodes, feature_embeddings)

            recommendations = matcher.get_expansion_recommendations(
                largest_dense_nodes, feature_embeddings
            )
            result["expansion_recommendations"] = recommendations

            if os.path.exists(csv_path):
                os.remove(csv_path)

            serialized_result = self._serialize_numpy(result)

            self.update_job_status(
                job_id, "completed", "Analysis complete", serialized_result
            )
            await notification_service.broadcast_job_update(
                job_id,
                {
                    "status": "completed",
                    "progress": "Analysis complete",
                    "result": serialized_result,
                },
            )

        except Exception as e:
            self.update_job_status(job_id, "failed", f"Error: {str(e)}", error=str(e))

            await notification_service.broadcast_job_update(
                job_id,
                {"status": "failed", "error": str(e), "progress": f"Error: {str(e)}"},
            )

            if os.path.exists(csv_path):
                os.remove(csv_path)


class FileService:
    """Service for handling file operations"""

    @staticmethod
    async def save_uploaded_file(file) -> str:
        """Save uploaded file to temporary location"""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            return tmp_file.name

    @staticmethod
    def validate_csv_file(filename: str) -> bool:
        """Validate if file is CSV"""
        return filename.lower().endswith(".csv")


analysis_service = AnalysisService()
file_service = FileService()
