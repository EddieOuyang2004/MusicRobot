"""Extract cited paper figures as vector-preserving PDF crops.

Source PDFs are downloaded separately into ``tmp/pdfs/literature_sources``.
Crop boxes use PDF points with the origin at the lower-left of each page.
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "tmp" / "pdfs" / "literature_sources"
OUTPUT_DIR = ROOT / "docs" / "thesis" / "figures"


# source file, one-based page number, (left, bottom, right, top), output file
FIGURES = (
    (
        "aistpp_fact.pdf",
        1,
        (45, 475, 555, 615),
        "chapter2_aistpp_fact_overview.pdf",
    ),
    (
        "beatit.pdf",
        1,
        (125, 345, 485, 470),
        "chapter2_beatit_conditioning.pdf",
    ),
    (
        "gmr.pdf",
        6,
        (45, 122, 305, 270),
        "chapter2_retargeting_gap.pdf",
    ),
    (
        "discoforcing.pdf",
        3,
        (55, 510, 555, 730),
        "chapter2_discoforcing_streaming.pdf",
    ),
)


def extract(source_name: str, page_number: int, box: tuple[int, int, int, int], output_name: str) -> None:
    reader = PdfReader(SOURCE_DIR / source_name)
    page = reader.pages[page_number - 1]
    rectangle = RectangleObject(box)
    page.mediabox = rectangle
    page.cropbox = rectangle
    page.trimbox = rectangle
    page.artbox = rectangle
    page.bleedbox = rectangle

    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": output_name,
            "/Subject": f"Figure crop from {source_name}, page {page_number}",
        }
    )
    with (OUTPUT_DIR / output_name).open("wb") as output:
        writer.write(output)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for args in FIGURES:
        extract(*args)


if __name__ == "__main__":
    main()
