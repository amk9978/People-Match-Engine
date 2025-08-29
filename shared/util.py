import logging
import sys
from typing import Dict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sanitize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Remove NaN, inf, and other non-JSON-serializable float values from metrics"""
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
            sanitized[key] = float(value)
        else:
            logger.warning(f"Removed non-serializable metric {key}: {value}")
    return sanitized


def serialize_numpy(obj):
    """Convert numpy objects and other non-JSON types to JSON serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Removed non-serializable numpy float: {value}")
            return None
        return value
    elif isinstance(obj, (int, float)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            logger.warning(f"Removed non-serializable float: {obj}")
            return None
        return obj
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(item) for item in obj]
    return obj
