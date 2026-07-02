import json
import asyncio
import os
from typing import List, Dict, Any
from datetime import datetime
from app.agent import get_agent_response
from app.embedding_client import embedding_service
from app.pinecone_service import pinecone_service
from logger.logger import get_logger

logger = get_logger(__name__)


class PrecisionEvaluator:
    def __init__(self, samples_file: str = "data/test_samples.json"):
        self.samples_file = samples_file
        self.samples = []
        self.results = []
        
    def load_samples(self):
        if not os.path.exists(self.samples_file):
            raise FileNotFoundError(f"Samples file not found: {self.samples_file}")
        
        with open(self.samples_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.samples = data.get("samples", [])
        
        logger.info(f"Loaded {len(self.samples)} test samples")
        return self.samples

    def calculate_precision(self, recommended: List[str], expected: List[str]) -> float:
        if not recommended:
            return 0.0
        
        correct = sum(1 for rec in recommended if rec in expected)
        return correct / len(recommended)

    def calculate_recall(self, recommended: List[str], expected: List[str]) -> float:
        if not expected:
            return 1.0
        
        correct = sum(1 for rec in recommended if rec in expected)
        return correct / len(expected)

    def calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def validate_schema(self, response: Dict) -> bool:
        required_keys = ["reply", "recommendations", "end_of_conversation"]
        
        for key in required_keys:
            if key not in response:
                return False
        
        if not isinstance(response["recommendations"], list):
            return False
        
        return True

    async def evaluate_sample(self, sample: Dict) -> Dict:
        sample_name = sample.get("name", "Unknown")
        turns = sample.get("turns", [])
        expected_names = sample.get("expected_names", [])
        expected_count = sample.get("expected_count", 0)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {sample_name}")
        logger.info(f"{'='*60}")
        
        messages = []
        final_response = None
        
        for turn in turns:
            messages.append(turn)
            
            try:
                response = await get_agent_response(messages)
                final_response = response
                
                if response.get("recommendations"):
                    logger.info(f"Found {len(response['recommendations'])} recommendations")
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "sample_name": sample_name,
                    "success": False,
                    "error": str(e)
                }
        
        if not final_response:
            return {
                "sample_name": sample_name,
                "success": False,
                "error": "No response received"
            }
        
        recommendations = final_response.get("recommendations", [])
        recommended_names = [r.get("name", "") for r in recommendations]
        
        precision = self.calculate_precision(recommended_names, expected_names)
        recall = self.calculate_recall(recommended_names, expected_names)
        f1 = self.calculate_f1(precision, recall)
        
        schema_valid = self.validate_schema(final_response)
        
        success = precision >= 0.7 and schema_valid
        
        result = {
            "sample_name": sample_name,
            "success": success,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "recommended_count": len(recommended_names),
            "expected_count": expected_count,
            "recommended_names": recommended_names,
            "expected_names": expected_names,
            "correct_recommendations": [r for r in recommended_names if r in expected_names],
            "incorrect_recommendations": [r for r in recommended_names if r not in expected_names],
            "missing_recommendations": [r for r in expected_names if r not in recommended_names],
            "schema_valid": schema_valid,
            "end_of_conversation": final_response.get("end_of_conversation", False)
        }
        
        self._print_result(result)
        
        return result

    def _print_result(self, result: Dict):
        status = "✅" if result["success"] else "❌"
        print(f"\n{status} {result['sample_name']}")
        print(f"   Precision: {result['precision']:.3f}")
        print(f"   Recall: {result['recall']:.3f}")
        print(f"   F1 Score: {result['f1']:.3f}")
        print(f"   Recommendations: {result['recommended_count']}/{result['expected_count']}")
        
        if result["correct_recommendations"]:
            print(f"   Correct: {result['correct_recommendations']}")
        
        if result["incorrect_recommendations"]:
            print(f"   Incorrect: {result['incorrect_recommendations']}")
        
        if result["missing_recommendations"]:
            print(f"   Missing: {result['missing_recommendations']}")
        
        print(f"   Schema Valid: {result['schema_valid']}")

    async def evaluate_all(self) -> Dict:
        self.load_samples()
        
        logger.info("\n" + "="*60)
        logger.info("PRECISION EVALUATION")
        logger.info("="*60)
        
        total = len(self.samples)
        success_count = 0
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for idx, sample in enumerate(self.samples, 1):
            logger.info(f"\n[{idx}/{total}] Processing...")
            result = await self.evaluate_sample(sample)
            self.results.append(result)
            
            if result["success"]:
                success_count += 1
            
            all_precisions.append(result["precision"])
            all_recalls.append(result["recall"])
            all_f1s.append(result["f1"])
        
        avg_precision = sum(all_precisions) / total if total > 0 else 0
        avg_recall = sum(all_recalls) / total if total > 0 else 0
        avg_f1 = sum(all_f1s) / total if total > 0 else 0
        success_rate = (success_count / total) * 100 if total > 0 else 0
        
        summary = {
            "total_samples": total,
            "successful": success_count,
            "failed": total - success_count,
            "success_rate": success_rate,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "detailed_results": self.results
        }
        
        self._print_summary(summary)
        
        return summary

    def _print_summary(self, summary: Dict):
        print("\n" + "="*60)
        print("PRECISION EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nOverall Results:")
        print(f"  Total Samples: {summary['total_samples']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\nPrecision & Recall:")
        print(f"  Average Precision: {summary['avg_precision']:.3f}")
        print(f"  Average Recall: {summary['avg_recall']:.3f}")
        print(f"  Average F1 Score: {summary['avg_f1']:.3f}")
        
        score = summary['avg_f1']
        if score >= 0.9:
            grade = "A - Excellent"
        elif score >= 0.8:
            grade = "B - Good"
        elif score >= 0.7:
            grade = "C - Acceptable"
        else:
            grade = "D - Needs Improvement"
        
        print(f"\nOverall Grade: {grade}")
        print("="*60)

    def export_results(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"precision_results_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results exported to: {filename}")


async def main():
    await embedding_service.initialize()
    pinecone_service.initialize()
    
    evaluator = PrecisionEvaluator("data/test_samples.json")
    results = await evaluator.evaluate_all()
    evaluator.export_results()


if __name__ == "__main__":
    asyncio.run(main())