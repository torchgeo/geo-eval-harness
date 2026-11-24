import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "page",
    [
        "docs/_static/results-explorer.html",
        "docs/_static/ranking_explorer.html",
        "scripts/ranking_explorer.template.html",
    ],
)
def test_explorer_labels_escape_markup_and_attributes(page: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the page's JavaScript formatter")
    source = (Path(__file__).resolve().parents[1] / page).read_text(encoding="utf-8")
    helper = re.search(r"function escapeHTML\(value\) \{.*?\n\s*\}", source, re.DOTALL)
    assert helper is not None
    label = """"><img src=x onerror='alert(1)'>& northern tile"""
    script = tmp_path / "format-label.js"
    script.write_text(
        helper.group() + f"\nprocess.stdout.write(escapeHTML({json.dumps(label)}));",
        encoding="utf-8",
    )
    result = subprocess.run([node, str(script)], capture_output=True, encoding="utf-8", check=True)
    assert html.unescape(result.stdout) == label
    assert not any(character in result.stdout for character in "<>\"'")
