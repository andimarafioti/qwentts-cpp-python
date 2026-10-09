from html.parser import HTMLParser
import sys

import pytest

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


def test_rocm_only_publication_preserves_published_flavors_without_copying_payloads(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    candidate = "qwentts_cpp_python-0.5.0+rocm724-py3-none-linux_x86_64.whl"
    (dist / candidate).write_bytes(b"new ROCm wheel")
    published = [
        "whl/cpu/qwentts_cpp_python-0.4.0+cpu-py3-none-manylinux_2_35_x86_64.whl",
        "whl/cpu/qwentts_cpp_python-0.5.0+cpu-py3-none-manylinux_2_35_x86_64.whl",
        "whl/cpu/qwentts_cpp_python-0.5.0+cpu-py3-none-manylinux_2_35_aarch64.whl",
        "whl/cu128/qwentts_cpp_python-0.5.0+cu128-py3-none-manylinux_2_39_x86_64.whl",
        "whl/metal/qwentts_cpp_python-0.3.1+metal-py3-none-macosx_14_0_arm64.whl",
        "whl/rocm724/qwentts_cpp_python-0.4.0+rocm724-py3-none-linux_x86_64.whl",
        "README.md",
    ]

    class MetadataOnlyApi:
        def list_repo_files(self, repo_id, *, repo_type):
            assert (repo_id, repo_type) == ("owner/wheels", "dataset")
            return published

    monkeypatch.setattr("huggingface_hub.HfApi", MetadataOnlyApi)
    out = tmp_path / "upload"
    monkeypatch.setattr(sys, "argv", ["prepare_hf_wheel_repo", "--dist", str(dist), "--out", str(out),
                                     "--repo-id", "owner/wheels", "--preserve-existing"])
    assert main() == 0
    assert [p.name for p in out.rglob("*.whl")] == [candidate]
    pages = {page.stem: page for page in (out / "whl").glob("*.html") if page.stem != "index"}
    assert set(pages) == {"cpu", "cu128", "metal", "rocm724"}
    assert len(Links(pages["cpu"].read_text()).hrefs) == 2
    assert "0.4.0" not in pages["cpu"].read_text()
    assert "0.4.0" not in pages["rocm724"].read_text()
    for flavor in ("cpu", "cu128", "metal"):
        assert all("whl/" + href in published for href in Links(pages[flavor].read_text()).hrefs)
    assert "qwentts-cpp-python==0.3.1+metal" in (out / "README.md").read_text()


def test_remote_metadata_failure_leaves_prepared_index_untouched(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "qwentts_cpp_python-0.5.0+rocm724-py3-none-linux_x86_64.whl").touch()
    out = tmp_path / "upload"
    out.mkdir()
    previous = out / "index.html"
    previous.write_text("previous index")

    class UnavailableApi:
        def list_repo_files(self, *_args, **_kwargs):
            raise ConnectionError("Hub metadata unavailable")

    monkeypatch.setattr("huggingface_hub.HfApi", UnavailableApi)
    monkeypatch.setattr(sys, "argv", ["prepare_hf_wheel_repo", "--dist", str(dist), "--out", str(out),
                                     "--repo-id", "owner/wheels", "--preserve-existing"])
    with pytest.raises(ConnectionError, match="Hub metadata unavailable"):
        main()
    assert previous.read_text() == "previous index"
