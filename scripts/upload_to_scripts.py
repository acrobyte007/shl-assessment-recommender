import os
import json
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from app.embedding_client import embedding_service
from app.pinecone_service import pinecone_service
from scripts.catalog_loader import catalog_loader

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self.namespace = "shl_catalog"
        self.batch_size = 50

    async def run(self):
        logger.info("=" * 60)
        logger.info("STARTING PIPELINE: Catalog → Embeddings → Pinecone")
        logger.info("=" * 60)

        await embedding_service.initialize()

        logger.info("Loading catalog...")
        catalog_loader.load()
        
        stats = catalog_loader.get_catalog_stats()
        logger.info(f"Catalog stats: {json.dumps(stats, indent=2)}")

        active_items = catalog_loader.get_active_items()
        logger.info(f"Processing {len(active_items)} active assessments")

        if not active_items:
            logger.error("No active items found. Exiting.")
            return

        searchable_texts = catalog_loader.get_searchable_texts(active_only=True)
        metadata_list = catalog_loader.get_all_metadata(active_only=True)

        logger.info(f"Generating embeddings for {len(searchable_texts)} items...")
        embedding_results = embedding_service.embed(searchable_texts)

        logger.info("Preparing vectors for Pinecone...")
        assessments = []
        
        for idx, (item, metadata) in enumerate(zip(active_items, metadata_list)):
            embedding_data = embedding_results.get(idx)
            
            if not embedding_data:
                logger.warning(f"No embedding for item {idx}: {item.get('name')}")
                continue

            assessments.append({
                "id": metadata["id"],
                "values": embedding_data["embedding"],
                "metadata": {
                    "entity_id": metadata.get("entity_id", ""),
                    "name": metadata.get("name", ""),
                    "link": metadata.get("link", ""),
                    "keys": metadata.get("keys", ""),
                    "job_levels": metadata.get("job_levels", ""),
                    "remote": metadata.get("remote", "no"),
                    "adaptive": metadata.get("adaptive", "no"),
                    "duration": metadata.get("duration", ""),
                    "languages": metadata.get("languages", ""),
                    "description": metadata.get("description", "")
                }
            })

        logger.info(f"Uploading {len(assessments)} assessments to Pinecone...")
        
        result = pinecone_service.upsert_assessments(
            namespace=self.namespace,
            assessments=assessments
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total vectors uploaded: {result['total_vectors']}")
        logger.info(f"Batches processed: {result['batches']}")
        logger.info(f"Namespace: {result['namespace']}")
        logger.info("=" * 60)

        stats = pinecone_service.get_index_stats()

    async def run_incremental(self, assessment_ids: List[str]):
        logger.info("Running incremental upload...")
        
        await embedding_service.initialize()
        catalog_loader.load()

        items_to_upload = []
        for assessment_id in assessment_ids:
            item = catalog_loader.get_by_id(assessment_id)
            if item and item.get("status") == "ok":
                items_to_upload.append(item)
            else:
                logger.warning(f"Item {assessment_id} not found or inactive")

        if not items_to_upload:
            logger.error("No valid items to upload")
            return

        searchable_texts = [catalog_loader.create_searchable_text(item) for item in items_to_upload]
        embedding_results = embedding_service.embed(searchable_texts)

        assessments = []
        for idx, item in enumerate(items_to_upload):
            embedding_data = embedding_results.get(idx)
            if not embedding_data:
                continue

            assessments.append({
                "id": f"item_{item.get('entity_id')}",
                "values": embedding_data["embedding"],
                "metadata": {
                    "entity_id": item.get("entity_id", ""),
                    "name": item.get("name", ""),
                    "link": item.get("link", ""),
                    "keys": ",".join(item.get("keys", [])),
                    "job_levels": ",".join(item.get("job_levels", [])),
                    "remote": item.get("remote", "no"),
                    "adaptive": item.get("adaptive", "no"),
                    "duration": item.get("duration", ""),
                    "languages": ",".join(item.get("languages", [])),
                    "description": item.get("description", "")
                }
            })

        result = pinecone_service.upsert_assessments(
            namespace=self.namespace,
            assessments=assessments
        )

        logger.info(f"Incremental upload complete: {result['total_vectors']} vectors")

    def delete_all(self):
        logger.info("Deleting all vectors from Pinecone...")
        pinecone_service.initialize()
        
        try:
            pinecone_service._index.delete(namespace=self.namespace, delete_all=True)
            logger.info(f"Deleted all vectors in namespace: {self.namespace}")
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise


async def main():
    pipeline = Pipeline()
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "delete":
            pipeline.delete_all()
            return
        elif sys.argv[1] == "stats":
            catalog_loader.load()
            print(json.dumps(catalog_loader.get_catalog_stats(), indent=2))
            return
    
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())