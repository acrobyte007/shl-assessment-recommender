import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatalogLoader:
    def __init__(self, data_path: str ="data/shl_product_catalog.json"):
        self.data_path = data_path
        self.items = []
        self.active_items = []
        self._loaded = False

    def load(self) -> List[Dict[str, Any]]:
        if self._loaded:
            return self.items

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Catalog file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.active_items = [item for item in self.items if item.get("status") == "ok"]
        self._loaded = True

        logger.info(f"Loaded {len(self.items)} items, {len(self.active_items)} active")
        return self.items

    def get_active_items(self) -> List[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        return self.active_items

    def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        
        for item in self.items:
            if item.get("entity_id") == entity_id:
                return item
        return None

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        
        for item in self.items:
            if item.get("name", "").lower() == name.lower():
                return item
        return None

    def get_keys(self) -> List[str]:
        if not self._loaded:
            self.load()
        
        keys_set = set()
        for item in self.active_items:
            for key in item.get("keys", []):
                keys_set.add(key)
        return sorted(list(keys_set))

    def get_job_levels(self) -> List[str]:
        if not self._loaded:
            self.load()
        
        levels_set = set()
        for item in self.active_items:
            for level in item.get("job_levels", []):
                levels_set.add(level)
        return sorted(list(levels_set))

    def create_searchable_text(self, item: Dict[str, Any]) -> str:
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            f"Keys: {', '.join(item.get('keys', []))}",
            f"Job Levels: {', '.join(item.get('job_levels', []))}"
        ]
        
        if item.get("remote") == "yes":
            parts.append("Remote Testing Available")
        
        if item.get("adaptive") == "yes":
            parts.append("Adaptive Test")
        else:
            parts.append("Non-adaptive Test")
        
        if item.get("duration"):
            parts.append(f"Duration: {item['duration']}")
        
        if item.get("languages"):
            parts.append(f"Languages: {', '.join(item.get('languages', []))}")
        
        return " | ".join(parts)

    def get_searchable_texts(self, active_only: bool = True) -> List[str]:
        items = self.get_active_items() if active_only else self.items
        return [self.create_searchable_text(item) for item in items]

    def get_all_metadata(self, active_only: bool = True) -> List[Dict[str, Any]]:
        items = self.get_active_items() if active_only else self.items
        
        metadata_list = []
        for idx, item in enumerate(items):
            metadata_list.append({
                "id": f"item_{idx}",
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
            })
        
        return metadata_list

    def get_assessment_ids(self, active_only: bool = True) -> List[str]:
        items = self.get_active_items() if active_only else self.items
        return [f"item_{i}" for i in range(len(items))]

    def get_catalog_stats(self) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        
        return {
            "total_items": len(self.items),
            "active_items": len(self.active_items),
            "inactive_items": len(self.items) - len(self.active_items),
            "unique_keys": len(self.get_keys()),
            "unique_job_levels": len(self.get_job_levels()),
            "remote_available": len([i for i in self.active_items if i.get("remote") == "yes"]),
            "adaptive_available": len([i for i in self.active_items if i.get("adaptive") == "yes"])
        }


catalog_loader = CatalogLoader()