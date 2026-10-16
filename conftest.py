"""
让 pytest 能 import 到 src 包。

项目根目录下放一个空的 conftest.py，pytest 就会把根目录加进 sys.path，
于是 `from src.agent.tools import ...` 在测试里可用 —— 和跑
`python -m src.runner` 时的导入方式保持一致。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
