"""让 tests/ 能 import 到 tools/ 下的模块。

`judge.py` 放在 tools/ 里（和 retrieval.py 等一起），测试要能 import 它。
不用装包，直接把路径加进去——这个项目不是 pip 包，不搞 setup.py。
"""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
