from html.parser import HTMLParser
import sys

from scripts.prepare_hf_wheel_repo import main


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.hrefs = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for name, value in attrs if name == "href")


def test_rocm_index_preserves_other_backends_and_documents_runtime(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    platforms = {
        "cpu": ["manylinux_2_35_x86_64"],
        "cu128": ["manylinux_2_35_x86_64", "manylinux_2_39_x86_64"],
        "metal": ["macosx_14_0_arm64"],
        "rocm724": ["linux_x86_64"],
    }
    for flavor, tags in platforms.items():
        for tag in tags:
            (dist / f"qwentts_cpp_python-0.5.0+{flavor}-py3-none-{tag}.whl").write_bytes(b"wheel fixture")
    out = tmp_path / "upload"
    monkeypatch.setattr(sys, "argv", ["prepare_hf_wheel_repo", "--dist", str(dist),
                                     "--out", str(out), "--repo-id", "owner/wheels"])
    assert main() == 0
    index = out / "whl" / "index.html"
    pages = [index.parent / href for href in Links(index.read_text()).hrefs]
    assert {page.stem for page in pages} == set(platforms)
    linked_wheels = [page.parent / href for page in pages for href in Links(page.read_text()).hrefs]
    assert {wheel.name for wheel in linked_wheels} == {wheel.name for wheel in dist.iterdir()}
    assert all(wheel.read_bytes() == b"wheel fixture" for wheel in linked_wheels)
    readme = (out / "README.md").read_text()
    assert "qwentts-cpp-python==0.5.0+rocm724" in readme
    assert "owner/wheels/tree/main/whl/rocm724" in readme
    for requirement in ("hipBLAS", "hipBLASLt", "rocBLAS/Tensile", "Ubuntu 24.04", "gfx942"):
        assert requirement in readme
