"""Convert pytest JUnit failures into GitHub Actions error annotations."""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree


def _escape(value: str) -> str:
    """Escape values for the GitHub Actions workflow command protocol."""

    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape(value).replace(",", "%2C").replace(":", "%3A")


def _failure_text(testcase: ElementTree.Element, failure: ElementTree.Element) -> str:
    message = failure.get("message", "").strip()
    detail = (failure.text or "").strip()
    name = testcase.get("name", "unknown test")
    classname = testcase.get("classname", "")
    test_name = f"{classname}.{name}" if classname else name
    summary = f"{test_name}: {message}" if message else test_name
    if detail and detail != message:
        summary = f"{summary}\n{detail}"
    return summary


def main(path: str) -> int:
    result_path = Path(path)
    if not result_path.exists():
        print(f"JUnit results not found: {result_path}", file=sys.stderr)
        return 0

    try:
        root = ElementTree.parse(result_path).getroot()
    except (ElementTree.ParseError, OSError) as error:
        print(f"Could not read JUnit results: {error}", file=sys.stderr)
        return 0

    count = 0
    for testcase in root.iter("testcase"):
        for failure in (*testcase.findall("failure"), *testcase.findall("error")):
            attributes = []
            source = testcase.get("file")
            if source:
                attributes.append(f"file={_escape_property(source)}")
            line = testcase.get("line")
            if line and line.isdigit():
                attributes.append(f"line={line}")
            title = _escape_property(
                f"pytest failure: {testcase.get('name', 'unknown test')}"
            )
            attributes.insert(0, f"title={title}")
            metadata = ",".join(attributes)
            print(f"::error {metadata}::{_escape(_failure_text(testcase, failure))}")
            count += 1

    print(f"JUnit annotations emitted: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "pytest-results.xml"))
