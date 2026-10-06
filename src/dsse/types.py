from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DSSEResult:
    url: str
    stage1_prediction_set: List[str]
    stage1_route: str
    final_prediction: str
    stage_used: int
    stage2_used: bool
    malicious_probability: float
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "stage1_prediction_set": self.stage1_prediction_set,
            "stage1_route": self.stage1_route,
            "final_prediction": self.final_prediction,
            "stage_used": self.stage_used,
            "stage2_used": self.stage2_used,
            "malicious_probability": self.malicious_probability,
            "latency_ms": self.latency_ms
        }
