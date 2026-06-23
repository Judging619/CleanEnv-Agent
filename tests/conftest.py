import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Smoke tests only verify API wiring and persistence behavior. They should not
# require real online model credentials during collection/import.
os.environ.setdefault("DASHSCOPE_API_KEY", "test_dashscope_api_key")
os.environ.setdefault("AMAP_WEB_SERVICE_KEY", "test_amap_web_service_key")
