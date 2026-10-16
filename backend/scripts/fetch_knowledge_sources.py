"""Download the curated extension fact sheets that seed the RAG knowledge base.

Usage (from backend/):
    python scripts/fetch_knowledge_sources.py            # download missing sources
    python scripts/fetch_knowledge_sources.py --refresh  # re-download everything
    python scripts/fetch_knowledge_sources.py --list     # print the source list, download nothing

Each source is a specific, publicly available university/government extension publication
(no crawling beyond one level: TNAU crop index pages are read to find their disease pages).
HTML pages are reduced to their article text and saved as .txt with a provenance header; PDFs are
saved as-is. Matching rows in knowledge/source_manifest.csv are written with a
SHA-256 hash; rows added by hand for other document IDs are preserved.

Downloaded files stay in git-ignored knowledge/source_docs/ and are used for non-commercial
academic research only. Review each source before relying on it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "knowledge" / "source_docs"
MANIFEST = ROOT / "knowledge" / "source_manifest.csv"
USER_AGENT = "Mozilla/5.0 (crop-disease-multimodal-ai academic research)"
USAGE_NOTE = (
    "Publicly available extension publication; local copy for non-commercial academic research; "
    "not redistributed; follow the publisher's terms"
)
FIELDS = [
    "document_id",
    "title",
    "organization",
    "url",
    "publication_date",
    "crop",
    "disease",
    "source_type",
    "license_or_usage_note",
    "local_file_path",
    "version_hash",
    "author",
]

NCSU = "NC State Extension"
OSU = "Ohio State University Extension (Ohioline)"


@dataclass(frozen=True)
class Source:
    document_id: str
    url: str
    organization: str
    crop: str
    disease: str
    title: str = ""  # required for PDFs; HTML titles are read from the page


SOURCES = [
    # Tomato
    Source(
        "NCSU_TOMATO_EARLY_BLIGHT",
        "https://content.ces.ncsu.edu/early-blight-of-tomato",
        NCSU,
        "Tomato",
        "Early Blight",
    ),
    Source(
        "NCSU_TOMATO_LATE_BLIGHT",
        "https://content.ces.ncsu.edu/tomato-late-blight",
        NCSU,
        "Tomato",
        "Late Blight",
    ),
    Source(
        "NCSU_TOMATO_SEPTORIA",
        "https://content.ces.ncsu.edu/septoria-leaf-spot-of-tomato",
        NCSU,
        "Tomato",
        "Septoria Leaf Spot",
    ),
    Source(
        "NCSU_BACTERIAL_SPOT_TOMATO_PEPPER",
        "https://content.ces.ncsu.edu/bacterial-spot-of-pepper-and-tomato",
        NCSU,
        "Tomato; Pepper",
        "Bacterial Spot",
    ),
    Source(
        "NCSU_TOMATO_TYLCV",
        "https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus",
        NCSU,
        "Tomato",
        "Tomato Yellow Leaf Curl Virus",
    ),
    Source(
        "NCSU_TOMATO_TOBAMOVIRUSES",
        "https://content.ces.ncsu.edu/tobamoviruses-that-affect-tomato-tmv-tomv-tobrfv",
        NCSU,
        "Tomato",
        "Tomato Mosaic Virus",
    ),
    Source(
        "NCSU_TOMATO_FOLIAR_FUNGAL",
        "https://content.ces.ncsu.edu/foliar-fungal-diseases-on-high-tunnel-and-greenhouse-tomatoes",
        NCSU,
        "Tomato",
        "Leaf Mold",
    ),
    Source(
        "NCSU_TWOSPOTTED_SPIDER_MITE",
        "https://content.ces.ncsu.edu/twospotted-spider-mite",
        NCSU,
        "Tomato",
        "Spider Mites (Two-spotted spider mite)",
    ),
    # Potato
    Source(
        "NCSU_POTATO_LATE_BLIGHT",
        "https://content.ces.ncsu.edu/potato-late-blight",
        NCSU,
        "Potato",
        "Late Blight",
    ),
    Source(
        "UF_EARLY_BLIGHT_TOMATO_POTATO",
        "https://www1.maine.gov/DACF/php/gotpests/diseases/factsheets/early-blight-fla.pdf",
        "University of Florida IFAS Extension (hosted by Maine DACF)",
        "Potato; Tomato",
        "Early Blight",
        title="Early Blight of Tomato and Potato",
    ),
    # Corn (maize)
    Source(
        "NCSU_CORN_RUSTS",
        "https://content.ces.ncsu.edu/corn-rusts-common-and-southern-rust",
        NCSU,
        "Corn (Maize)",
        "Common Rust",
    ),
    Source(
        "NCSU_CORN_GRAY_LEAF_SPOT",
        "https://content.ces.ncsu.edu/gray-leaf-spot-in-corn",
        NCSU,
        "Corn (Maize)",
        "Cercospora Leaf Spot (Gray Leaf Spot)",
    ),
    Source(
        "UDEL_NORTHERN_CORN_LEAF_BLIGHT",
        "https://www.udel.edu/content/dam/udelImages/canr/pdfs/extension/factsheets/"
        "Northern-Corn-Leaf-Blight.pdf",
        "University of Delaware Cooperative Extension",
        "Corn (Maize)",
        "Northern Leaf Blight",
        title="Northern Corn Leaf Blight",
    ),
    # Apple
    Source(
        "OSU_APPLE_SCAB",
        "https://ohioline.osu.edu/factsheet/plpath-fru-23",
        OSU,
        "Apple",
        "Apple Scab",
    ),
    Source(
        "NCSU_CEDAR_APPLE_RUST",
        "https://content.ces.ncsu.edu/cedar-apple-rusts",
        NCSU,
        "Apple",
        "Cedar Apple Rust",
    ),
    # Grape
    Source(
        "OSU_GRAPE_BLACK_ROT",
        "https://ohioline.osu.edu/factsheet/plpath-fru-24",
        OSU,
        "Grape",
        "Black Rot",
    ),
    Source(
        "UMAINE_GRAPE_BLACK_ROT",
        "https://extension.umaine.edu/ipm/ipddl/publications/5106e/",
        "University of Maine Cooperative Extension",
        "Grape",
        "Black Rot",
    ),
    Source(
        "OKSTATE_GRAPE_BLACK_ROT",
        "https://pods.okstate.edu/fact-sheets/EPP-7643web-color.pdf",
        "Oklahoma State University Extension",
        "Grape",
        "Black Rot",
        title="Black Rot of Grapes (EPP-7643)",
    ),
    # Peach, strawberry, squash, orange
    Source(
        "OSU_STONE_FRUIT_BACTERIAL_SPOT",
        "https://ohioline.osu.edu/factsheet/plpath-fru-38",
        OSU,
        "Peach",
        "Bacterial Spot",
    ),
    Source(
        "NCSU_STRAWBERRY_LEAF_SCORCH",
        "https://content.ces.ncsu.edu/leaf-scorch-of-strawberry",
        NCSU,
        "Strawberry",
        "Leaf Scorch",
    ),
    Source(
        "OSU_STRAWBERRY_LEAF_DISEASES",
        "https://ohioline.osu.edu/factsheet/plpath-fru-35",
        OSU,
        "Strawberry",
        "Leaf Scorch",
    ),
    Source(
        "NCSU_CUCURBIT_POWDERY_MILDEW",
        "https://content.ces.ncsu.edu/cucurbit-powdery-mildew",
        NCSU,
        "Squash",
        "Powdery Mildew",
    ),
    Source(
        "UF_CITRUS_HLB",
        "https://journals.flvc.org/edis/article/download/114966/113289",
        "University of Florida IFAS Extension (EDIS)",
        "Orange",
        "Huanglongbing (Citrus Greening)",
        title="Citrus Diseases Exotic to Florida: Huanglongbing (Citrus Greening) (PP-210, 2005)",
    ),
    Source(
        "WSU_CHERRY_POWDERY_MILDEW",
        "https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
        "Washington State University Tree Fruit Extension",
        "Cherry",
        "Powdery Mildew",
    ),
]

# More NC State Extension fact sheets: (slug, crop, disease)
_NCSU_EXTRA = [
    ("bacterial-canker-of-tomato", "Tomato", "Bacterial Canker"),
    ("bacterial-wilt-of-tomatoes", "Tomato", "Bacterial Wilt"),
    ("botrytis-gray-mold-of-tomato", "Tomato", "Gray Mold"),
    ("collar-rot-and-alternaria-stem-canker-of-tomato", "Tomato", "Collar Rot"),
    ("fusarium-wilt-of-tomato", "Tomato", "Fusarium Wilt"),
    ("gray-leaf-spot-of-tomato", "Tomato", "Gray Leaf Spot"),
    ("tomato-pith-necrosis", "Tomato", "Pith Necrosis"),
    ("tomato-spotted-wilt-virus-on-tomato-and-pepper", "Tomato; Pepper", "Tomato Spotted Wilt"),
    ("verticillium-wilt-of-tomato-and-eggplant", "Tomato; Eggplant", "Verticillium Wilt"),
    ("southern-blight-of-vegetable-crops", "Tomato; Pepper; Potato", "Southern Blight"),
    ("anthracnose-of-pepper", "Pepper", "Anthracnose"),
    ("cercospora-leaf-spot-frogeye-leaf-spot-on-pepper", "Pepper", "Cercospora Leaf Spot"),
    ("phytophthora-blight-of-peppers", "Pepper", "Phytophthora Blight"),
    ("cucurbit-downy-mildew", "Squash", "Downy Mildew"),
    ("anthracnose-of-cucurbits", "Squash", "Anthracnose"),
    ("gummy-stem-blight-and-phoma-blight-on-cucurbits", "Squash", "Gummy Stem Blight"),
    ("apple-powdery-mildew", "Apple", "Powdery Mildew"),
    ("fire-blight", "Apple", "Fire Blight"),
    ("flyspeck-and-sooty-blotch", "Apple", "Flyspeck and Sooty Blotch"),
    ("glomerella-leaf-spot-and-fruit-rot", "Apple", "Glomerella Leaf Spot"),
    ("angular-leafspot-of-strawberry", "Strawberry", "Angular Leaf Spot"),
]
SOURCES += [
    Source(
        "NCSU_" + slug.upper().replace("-", "_")[:60],
        f"https://content.ces.ncsu.edu/{slug}",
        NCSU,
        crop,
        disease,
    )
    for slug, crop, disease in _NCSU_EXTRA
]

# UC IPM Pest Management Guidelines: crop path -> (crop label, [(page slug, disease)])
UCIPM = "University of California Statewide IPM Program (UC IPM)"
_UCIPM_PAGES = {
    "tomato": (
        "Tomato",
        [
            ("early-blight", "Early Blight"),
            ("late-blight", "Late Blight"),
            ("bacterial-spot", "Bacterial Spot"),
            ("bacterial-speck", "Bacterial Speck"),
            ("bacterial-canker", "Bacterial Canker"),
            ("alternaria-stem-canker", "Alternaria Stem Canker"),
            ("anthracnose", "Anthracnose"),
            ("black-mold", "Black Mold"),
            ("fusarium-wilt", "Fusarium Wilt"),
            ("verticillium-wilt", "Verticillium Wilt"),
            ("gray-mold", "Gray Mold"),
            ("white-mold", "White Mold"),
            ("southern-blight", "Southern Blight"),
            ("powdery-mildew-on-field-grown-tomatoes", "Powdery Mildew"),
            ("tobacco-mosaic", "Tomato Mosaic Virus"),
            ("tomato-spotted-wilt", "Tomato Spotted Wilt"),
            ("tomato-yellow-leaf-curl", "Tomato Yellow Leaf Curl Virus"),
            ("diseases-virus-symptoms-you-may-see", "Virus Symptoms"),
            ("tomato-russet-mite", "Russet Mite"),
            ("whiteflies", "Whiteflies"),
        ],
    ),
    "potato": (
        "Potato",
        [
            ("early-blight", "Early Blight"),
            ("late-blight", "Late Blight"),
            ("black-dot", "Black Dot"),
            ("common-scab", "Common Scab"),
            ("verticillium-wilt", "Verticillium Wilt"),
            ("white-mold", "White Mold"),
            ("bacterial-soft-rot-and-blackleg", "Blackleg and Soft Rot"),
            ("pink-rot", "Pink Rot"),
            ("leafroll", "Potato Leafroll Virus"),
            ("mosaic-diseases-caused-by-potyviruses", "Potato Virus Y / Mosaic"),
        ],
    ),
    "corn": (
        "Corn (Maize)",
        [
            ("common-rust", "Common Rust"),
            ("common-smut-boil-smut", "Common Smut"),
            ("fusarium-ear-rot", "Fusarium Ear Rot"),
            ("fusarium-stalk-rot", "Fusarium Stalk Rot"),
            ("maize-dwarf-mosaic", "Maize Dwarf Mosaic"),
            ("charcoal-rot", "Charcoal Rot"),
            ("spider-mites", "Spider Mites"),
        ],
    ),
    "apple": (
        "Apple",
        [
            ("apple-scab", "Apple Scab"),
            ("fire-blight", "Fire Blight"),
            ("powdery-mildew", "Powdery Mildew"),
            ("european-canker", "European Canker"),
            ("phytophthora-root-and-crown-rot", "Phytophthora Root and Crown Rot"),
        ],
    ),
    "grape": (
        "Grape",
        [
            ("powdery-mildew", "Powdery Mildew"),
            ("downy-mildew", "Downy Mildew"),
            ("esca-black-measles", "Esca (Black Measles)"),
            ("botrytis-bunch-rot", "Botrytis Bunch Rot"),
            ("eutypa-dieback", "Eutypa Dieback"),
            ("botryosphaeria-dieback", "Botryosphaeria Dieback"),
            ("phomopsis-cane-and-leafspot", "Phomopsis Cane and Leaf Spot"),
            ("pierces-disease", "Pierce's Disease"),
            ("summer-bunch-rot-sour-rot", "Sour Rot"),
            ("virus-diseases", "Virus Diseases"),
        ],
    ),
    "peach": (
        "Peach",
        [
            ("peach-leaf-curl", "Peach Leaf Curl"),
            ("brown-rot-blossom-and-twig-blight", "Brown Rot"),
            ("shot-hole-disease", "Shot Hole"),
            ("scab", "Peach Scab"),
            ("powdery-mildew", "Powdery Mildew"),
            ("rust", "Rust"),
            ("bacterial-canker", "Bacterial Canker"),
        ],
    ),
    "cherry": (
        "Cherry",
        [
            ("powdery-mildew", "Powdery Mildew"),
            ("bacterial-canker", "Bacterial Canker"),
            ("brown-rot-blossom-and-twig-blight", "Brown Rot"),
        ],
    ),
    "strawberry": (
        "Strawberry",
        [
            ("angular-leaf-spot", "Angular Leaf Spot"),
            ("anthracnose", "Anthracnose"),
            ("botrytis-fruit-rot", "Gray Mold"),
            ("common-leaf-spot", "Common Leaf Spot"),
            ("powdery-mildew", "Powdery Mildew"),
            ("strawberry-leaf-blotch", "Leaf Blotch"),
            ("verticillium-wilt", "Verticillium Wilt"),
        ],
    ),
    "citrus": (
        "Orange (Citrus)",
        [
            ("huanglongbing", "Huanglongbing (Citrus Greening)"),
            ("asian-citrus-psyllid", "Asian Citrus Psyllid"),
            ("anthracnose", "Anthracnose"),
            ("brown-rot", "Brown Rot"),
            ("septoria-spot", "Septoria Spot"),
            ("tristeza-disease-complex", "Citrus Tristeza Virus"),
            ("phytophthora-root-rot", "Phytophthora Root Rot"),
        ],
    ),
    "cucurbits": (
        "Squash",
        [
            ("powdery-mildew", "Powdery Mildew"),
            ("downy-mildew", "Downy Mildew"),
            ("angular-leaf-spot", "Angular Leaf Spot"),
            ("anthracnose", "Anthracnose"),
            ("squash-mosaic-virus", "Squash Mosaic Virus"),
            ("cucumber-mosaic", "Cucumber Mosaic Virus"),
        ],
    ),
}
SOURCES += [
    Source(
        f"UCIPM_{path.upper()}_{slug.upper().replace('-', '_')}"[:80],
        f"https://ipm.ucanr.edu/agriculture/{path}/{slug}/",
        UCIPM,
        crop,
        disease,
    )
    for path, (crop, pages) in _UCIPM_PAGES.items()
    for slug, disease in pages
]

# TNAU Agritech Portal (India) crop index pages; disease pages are discovered from each index.
TNAU = "Tamil Nadu Agricultural University (TNAU Agritech Portal)"
TNAU_BASE = "https://agritech.tnau.ac.in/crop_protection/"
TNAU_INDEXES = {
    "veg_tomato": "Tomato",
    "veg_potato": "Potato",
    "veg_chilli": "Pepper (Chilli)",
    "veg_brinjal": "Brinjal (Eggplant)",
    "veg_bhendi": "Bhendi (Okra)",
    "veg_onion": "Onion",
    "veg_cabbage": "Cabbage",
    "cereals_maize": "Corn (Maize)",
    "fruits_grapes": "Grape",
    "fruits_apple": "Apple",
    "fruits_citrus": "Orange (Citrus)",
    "fruits_peach": "Peach",
    "fruits_banana": "Banana",
    "fruits_mango": "Mango",
}

_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    "figure",
    "iframe",
    "select",
}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "section",
    "article",
    "ul",
    "ol",
    "table",
    "blockquote",
}


class _ArticleExtractor(HTMLParser):
    """Collect visible text, preferring the main article container when the page marks one."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_tag: str | None = None  # tag that opened the region being skipped
        self.skip_nesting = 0

    def handle_starttag(self, tag, attrs):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_nesting += 1
            return
        classes = dict(attrs).get("class") or ""
        if tag in _SKIP_TAGS or "modal" in classes.split():
            self.skip_tag, self.skip_nesting = tag, 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")
            elif re.fullmatch(r"h[1-6]", tag):
                self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if self.skip_tag is None and tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_nesting -= 1
                if self.skip_nesting == 0:
                    self.skip_tag = None
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_tag is None:
            self.parts.append(data)


def _main_container(page: str) -> str:
    """Slice out the article body using markers used by the publishers in SOURCES."""
    for start_pattern in (
        r'<div[^>]*itemprop="articleBody"',
        r'<div class="page-content"',
        r"<article[\s>]",
        r"<main[\s>]",
    ):
        match = re.search(start_pattern, page)
        if match:
            tail = page[match.start() :]
            for end_marker in ('<div class="related"', "<h1>Authors", "</article>", "</main>"):
                end = tail.find(end_marker)
                if end > 0:
                    return tail[:end]
            return tail
    return page


# Trailing boilerplate (author bios, keyword tags, legal footers) starts at the first of these.
_TAIL_MARKERS = (
    "\nAuthors\n",
    "\n- Keywords:",
    "\nTopics:\n",
    "\nInformation in this publication",
    "\nAcknowledgement",
    "\nUpdated on ",
    "\nHome | About Us",
)
_DATE_PATTERNS = (
    r"Reviewed/Revised:\s*([^\n]+)",
    r"Publication date:\s*([^\n]+)",
    r"\nDate:\s*\n+\s*(\d{2}/\d{2}/\d{4})",
    r"Text Updated:\s*([^\n]+)",
    r"Updated on ([A-Za-z]+ \d{4})",
)


def publication_date(text: str) -> str:
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def strip_tail(text: str) -> str:
    cut = len(text)
    for marker in _TAIL_MARKERS:
        index = text.find(marker, len(text) // 3)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].rstrip()


def html_to_text(page: str) -> str:
    extractor = _ArticleExtractor()
    extractor.feed(_main_container(page))
    text = unescape("".join(extractor.parts))
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if not line.startswith("Skip to ") and line != "-"]
    lines = _drop_leading_navigation(lines)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _drop_leading_navigation(lines: list[str]) -> list[str]:
    content = [line for line in lines if line]
    if content and content[0] == "On This Page" and len(content) > 1:
        # UC IPM: skip the in-page table of contents up to where its first heading repeats.
        first_heading = content[1]
        positions = [i for i, line in enumerate(lines) if line == first_heading]
        if len(positions) > 1:
            return lines[positions[1] :]
    for index, line in enumerate(lines[:40]):
        # TNAU: content starts at the breadcrumb, e.g. "Horticultural crops :: Vegetables:: Tomato".
        if "::" in line and "TNAU" not in line:
            return lines[index:]
    return lines


def html_title(page: str) -> str:
    for pattern in (
        r'<h1 class="title"[^>]*>(.*?)</h1>',
        r'<h2 class="archive-title">(.*?)</h2>',
        r"<h1[^>]*>(.*?)</h1>",
        r"<title>(.*?)</title>",
    ):
        match = re.search(pattern, page, flags=re.S)
        if not match:
            continue
        raw = re.sub(r"<[^>]+>", " ", unescape(match.group(1))).strip()
        # Drop the series name some publishers append after a wide gap or a pipe.
        title = re.split(r"\s{3,}| \| ", raw)[0].strip()
        if title and title.lower() != "authors":
            return re.sub(r"\s+", " ", title)
    return ""


def fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get_content_type()


def tnau_sources() -> list[Source]:
    """Read each TNAU crop index and return one Source per linked disease page."""
    sources: list[Source] = []
    seen: set[str] = set()
    for key, crop in TNAU_INDEXES.items():
        index_url = f"{TNAU_BASE}crop_prot_crop%20diseases_{key}.html"
        try:
            page = fetch(index_url)[0].decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"FAILED  TNAU index {key}: {exc}", file=sys.stderr)
            continue
        for href, label in re.findall(r'<a[^>]+href="([^"#]+\.html)"[^>]*>(.*?)</a>', page, re.S):
            name = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", label))).strip()
            # Keep this crop's disease pages; skip site navigation and links to other crops.
            crop_word = key.split("_", 1)[1][:4]
            folder = href.rsplit("/", 1)[0] if "/" in href else ""
            is_crop_page = (
                (not folder and not href.startswith("crop_prot"))
                or (folder and crop_word in folder and not folder.startswith(".."))
                or (href.startswith("crop_prot") and f"_{key}_" in href)
            )
            if not name or not is_crop_page or href in seen:
                continue
            seen.add(href)
            stem = href.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("crop_prot_crop diseases_", "")
            slug = re.sub(r"[^A-Z0-9]+", "_", stem.upper()).strip("_")
            url = TNAU_BASE + href.replace(" ", "%20")
            sources.append(Source(f"TNAU_{slug}", url, TNAU, crop, name))
        time.sleep(1)
    return sources


def all_sources() -> list[Source]:
    return SOURCES + tnau_sources()


def read_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST.is_file():
        return {}
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        return {row["document_id"]: row for row in csv.DictReader(stream) if row["document_id"]}


def write_manifest(rows: dict[str, dict[str, str]]) -> None:
    with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download existing files")
    parser.add_argument("--list", action="store_true", help="print sources and exit")
    arguments = parser.parse_args()
    sources = all_sources()
    if arguments.list:
        for source in sources:
            print(f"{source.document_id:60} {source.crop:22} {source.disease}")
        print(f"{len(sources)} sources")
        return 0
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest()
    failures: list[str] = []
    for source in sources:
        existing = rows.get(source.document_id)
        path = SOURCE_DIR / (existing["local_file_path"] if existing else "")
        if existing and path.is_file() and not arguments.refresh:
            print(f"skip    {source.document_id} (already downloaded)")
            continue
        try:
            body, content_type = fetch(source.url)
        except Exception as exc:  # network errors are reported, not fatal for the batch
            failures.append(f"{source.document_id}: {exc}")
            print(f"FAILED  {source.document_id}: {exc}", file=sys.stderr)
            continue
        retrieved = date.today().isoformat()
        if content_type == "application/pdf" or body[:5] == b"%PDF-":
            filename = f"{source.document_id}.pdf"
            (SOURCE_DIR / filename).write_bytes(body)
            title = source.title or source.document_id
            published = ""
        else:
            page = body.decode("utf-8", errors="replace")
            title = source.title or html_title(page) or source.document_id
            if title.startswith("TNAU Agritech Portal"):
                title = f"{source.crop}: {source.disease} (TNAU Agritech Portal)"
            full_text = html_to_text(page)
            published = publication_date(full_text)
            text = strip_tail(full_text)
            if len(text) < 300:
                failures.append(f"{source.document_id}: extracted only {len(text)} characters")
                print(f"FAILED  {source.document_id}: page had too little text", file=sys.stderr)
                continue
            filename = f"{source.document_id}.txt"
            header = (
                f"{title}\nSource: {source.organization}\nURL: {source.url}\n"
                f"Published/revised: {published or 'unknown'}\nRetrieved: {retrieved}\n\n"
            )
            (SOURCE_DIR / filename).write_text(header + text + "\n", encoding="utf-8")
        digest = hashlib.sha256((SOURCE_DIR / filename).read_bytes()).hexdigest()
        rows[source.document_id] = {
            "document_id": source.document_id,
            "title": title,
            "organization": source.organization,
            "url": source.url,
            "publication_date": published,
            "crop": source.crop,
            "disease": source.disease,
            "source_type": "extension_factsheet",
            "license_or_usage_note": f"{USAGE_NOTE}; retrieved {retrieved}",
            "local_file_path": filename,
            "version_hash": digest,
            "author": "",
        }
        print(f"saved   {source.document_id} -> {filename} ({title})")
        time.sleep(1)  # be polite to publisher servers
    write_manifest(rows)
    print(f"\nManifest rows: {len(rows)}. Failures: {len(failures)}.")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
