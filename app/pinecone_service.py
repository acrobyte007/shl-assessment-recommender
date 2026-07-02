import os
import json
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, PineconeException
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeService:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self._pc = None
        self._index = None
        self._initialized = False
        self.index_name = "index"

    def initialize(self):
        if self._initialized:
            return

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise Exception("Missing PINECONE_API_KEY")

        self._pc = Pinecone(api_key=api_key)
        
        if self.index_name not in self._pc.list_indexes().names():
            raise Exception(f"Index '{self.index_name}' does not exist. Run upload script first.")
        
        self._index = self._pc.Index(self.index_name)
        self._initialized = True
        logger.info(f"Pinecone initialized with index: {self.index_name}")

    def upsert_assessments(
        self,
        namespace: str,
        assessments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        self.initialize()

        if not assessments:
            raise ValueError("Assessments list is empty")

        vectors_to_upsert = []
        for assessment in assessments:
            if "id" not in assessment or "values" not in assessment:
                raise ValueError("Each assessment must have 'id' and 'values'")
            
            vectors_to_upsert.append({
                "id": assessment["id"],
                "values": assessment["values"],
                "metadata": assessment.get("metadata", {})
            })

        responses = []
        for i in range(0, len(vectors_to_upsert), self.batch_size):
            batch = vectors_to_upsert[i:i + self.batch_size]
            try:
                res = self._index.upsert(namespace=namespace, vectors=batch)
                responses.append(res)
                logger.info(f"Uploaded batch {i//self.batch_size + 1}: {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Upsert failed: {e}")
                raise

        return {
            "total_vectors": len(vectors_to_upsert),
            "batches": len(responses),
            "namespace": namespace
        }

    def search_assessments(
        self,
        namespace: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Searching assessments in namespace '{namespace}' with top_k={top_k}")
        try:
            result = self._index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            logger.info(f"Search result: {result}")
            matches = result.get("matches", [])

            assessments = []
            for match in matches:
                metadata = match.get("metadata", {})
                assessments.append({
                    "id": match.get("id", ""),
                    "score": match.get("score", 0.0),
                    "name": metadata.get("name", ""),
                    "link": metadata.get("link", ""),
                    "keys": metadata.get("keys", "").split(",") if metadata.get("keys") else [],
                    "job_levels": metadata.get("job_levels", "").split(",") if metadata.get("job_levels") else [],
                    "remote": metadata.get("remote", "no"),
                    "adaptive": metadata.get("adaptive", "no"),
                    "duration": metadata.get("duration", ""),
                    "description": metadata.get("description", "")
                })

            return {
                "assessments": assessments,
                "total_matches": len(assessments),
                "namespace": namespace
            }

        except PineconeException as e:
            logger.error(f"Pinecone search error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            raise



pinecone_service = PineconeService()