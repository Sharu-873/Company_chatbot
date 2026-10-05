import os
import re
import time
import io
import pickle
import hashlib

from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    unquote,
)

import requests
import numpy as np
import faiss

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings.fastembed import (
    FastEmbedEmbeddings
)



# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://novatechrobo.com/"
MAX_DEPTH = 3
MAX_PAGES =300

DATA_DIR = "data"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

MIN_CHUNK_LENGTH = 80

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

BAD_URL_PATTERNS = [
    "privacy",
    "terms",
    "cookie",
    "login",
    "signup",
]

SKIP_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".zip",
    ".rar",
    ".mp4",
    ".mp3",
)

# Documents that should be processed separately
DOCUMENT_EXTENSIONS = (
    ".pdf",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

REMOVE_TAGS = {
    "script",
    "style",
    "nav",
    "header",
    "noscript",
    "svg",
    "form",
    "iframe",
    "aside",
}

CONTENT_SELECTORS = [
    "main",
    "article",
    '[role="main"]',
    ".content",
    ".main-content",
    ".page-content",
    "#content",
    "#main",
]
# =========================================================
# GLOBALS
# =========================================================

visited = set()
pages = []
seen_chunks = set()

# Normalize the base domain once
BASE_DOMAIN = (
    urlparse(BASE_URL)
    .netloc
    .lower()
    .replace("www.", "")
)


# =========================================================
# SESSION WITH RETRIES
# =========================================================

session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
    respect_retry_after_header=True,
)

adapter = HTTPAdapter(max_retries=retries)

session.mount("http://", adapter)
session.mount("https://", adapter)


# =========================================================
# URL HELPERS
# =========================================================

def normalize_url(url):
    """
    Normalize URLs so the crawler does not treat equivalent URLs
    as separate pages.

    Examples:
        https://www.novatechrobo.com/page/
        https://novatechrobo.com/page/
        https://novatechrobo.com/page/#section

    become the same canonical URL.
    """

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower().replace("www.", "")

    # Normalize path
    path = parsed.path or "/"

    # Remove duplicate trailing slash except for homepage
    if path != "/":
        path = path.rstrip("/")

    # We intentionally remove query parameters.
    # This prevents tracking URLs such as:
    # ?utm_source=...
    # ?utm_medium=...
    #
    # If the website later uses meaningful query parameters,
    # we can whitelist them.
    query = ""

    # Fragments (#section) are not separate documents
    fragment = ""

    return urlunparse((
        scheme,
        netloc,
        path,
        "",
        query,
        fragment,
    ))


def is_document_url(url):
    """
    Return True if the URL points to a document that should be
    processed separately from normal HTML pages.
    """

    path = urlparse(url).path.lower()

    return any(
        path.endswith(ext)
        for ext in DOCUMENT_EXTENSIONS
    )


def is_valid_url(url):
    """
    Check whether a URL is allowed to enter the crawler queue.
    """

    parsed = urlparse(url)

    # -----------------------------------------------------
    # 1. Only HTTP/HTTPS
    # -----------------------------------------------------

    if parsed.scheme.lower() not in {"http", "https"}:
        return False

    # -----------------------------------------------------
    # 2. Normalize and check domain
    # -----------------------------------------------------

    domain = (
        parsed.netloc
        .lower()
        .replace("www.", "")
    )

    if domain != BASE_DOMAIN:
        return False

    # -----------------------------------------------------
    # 3. Get clean URL/path
    # -----------------------------------------------------

    normalized = normalize_url(url)

    path = urlparse(normalized).path.lower()

    # -----------------------------------------------------
    # 4. Skip unwanted file types
    # -----------------------------------------------------

    # PDFs are intentionally NOT in SKIP_EXTENSIONS.
    # They will be handled separately by the PDF processor.
    if any(
        path.endswith(ext)
        for ext in SKIP_EXTENSIONS
    ):
        return False

    # -----------------------------------------------------
    # 5. Skip unwanted URL patterns
    # -----------------------------------------------------

    if any(
        bad.lower() in normalized.lower()
        for bad in BAD_URL_PATTERNS
    ):
        return False

    return True


# =========================================================
# TEXT CLEANING
# =========================================================

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Clean extracted webpage/PDF text while preserving information
    that may be useful for RAG retrieval.

    Important:
    - Do NOT remove all non-ASCII characters.
    - Do NOT remove meaningful phrases such as "Contact Us".
    - Preserve punctuation because it helps semantic retrieval.
    """

    if not text:
        return ""

    # -----------------------------------------------------
    # Normalize common whitespace characters
    # -----------------------------------------------------

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # -----------------------------------------------------
    # Remove control characters
    # -----------------------------------------------------

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text,
    )

    # -----------------------------------------------------
    # Normalize whitespace
    # -----------------------------------------------------

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # -----------------------------------------------------
    # Remove obvious repeated navigation/CTA junk
    #
    # IMPORTANT:
    # Do not remove "Contact Us" because contact information
    # is useful knowledge for the company RAG chatbot.
    # -----------------------------------------------------

    remove_patterns = [
        r"\bLearn More\b",
        r"\bWatch Video\b",
    ]

    for pattern in remove_patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # -----------------------------------------------------
    # Normalize excessive spaces again after removals
    # -----------------------------------------------------

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# =========================================================
# CHUNK DEDUPLICATION
# =========================================================

def normalize_for_hash(text):
    """
    Normalize text before hashing so insignificant differences
    do not create duplicate chunks.
    """

    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_hash(text):
    """
    Generate a stable hash for duplicate-chunk detection.
    """

    normalized = normalize_for_hash(text)

    return hashlib.md5(
        normalized.encode("utf-8")
    ).hexdigest()


# =========================================================
# HTML SCRAPER
# =========================================================

def scrape(url):
    """
    Fetch and extract useful content from an HTML page.

    PDFs and other document types are handled separately.
    """

    try:
        response = session.get(
            url,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # Content type
        # ----------------------------------------------------

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        # Only process HTML here.
        # PDF/document processing will be handled separately.
        if "text/html" not in content_type:
            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # ----------------------------------------------------
        # Metadata BEFORE removing elements
        # ----------------------------------------------------

        title = ""

        if soup.title:
            title = soup.title.get_text(
                " ",
                strip=True,
            )

        description = ""

        meta = soup.find(
            "meta",
            attrs={"name": "description"},
        )

        if meta:
            description = meta.get(
                "content",
                "",
            ).strip()

        # ----------------------------------------------------
        # Extract useful footer information BEFORE removing
        # footer/header elements.
        #
        # This is important because company contact details
        # may live in the footer.
        # ----------------------------------------------------

        footer_text = ""

        footer = soup.find("footer")

        if footer:
            footer_text = footer.get_text(
                "\n",
                strip=True,
            )

        # ----------------------------------------------------
        # Remove unwanted elements
        # ----------------------------------------------------

        for tag in soup.find_all(list(REMOVE_TAGS)):
            tag.decompose()

        # ----------------------------------------------------
        # Remove common boilerplate
        # ----------------------------------------------------

        boilerplate_selectors = [
            ".sidebar",
            ".breadcrumbs",
            ".breadcrumb",
            ".cookie-banner",
            ".cookie-consent",
            ".newsletter",
            ".advertisement",
            ".ads",
            ".pagination",
            ".social-share",
            ".share-buttons",
        ]

        for selector in boilerplate_selectors:
            for node in soup.select(selector):
                node.decompose()

        # ----------------------------------------------------
        # Locate main content
        # ----------------------------------------------------

        content = None

        for selector in CONTENT_SELECTORS:
            candidate = soup.select_one(selector)

            if candidate:
                # Ignore empty containers
                candidate_text = candidate.get_text(
                    " ",
                    strip=True,
                )

                if candidate_text:
                    content = candidate
                    break

        # Fallback to body/document
        if content is None:
            content = soup.body or soup

        # ----------------------------------------------------
        # Extract headings BEFORE converting to plain text
        # ----------------------------------------------------

        headings = {
            "h1": [
                h.get_text(" ", strip=True)
                for h in content.find_all("h1")
            ],
            "h2": [
                h.get_text(" ", strip=True)
                for h in content.find_all("h2")
            ],
            "h3": [
                h.get_text(" ", strip=True)
                for h in content.find_all("h3")
            ],
        }

        # Remove duplicate headings while preserving order
        for level in headings:
            headings[level] = list(
                dict.fromkeys(
                    h for h in headings[level]
                    if h
                )
            )

        # ----------------------------------------------------
        # Extract main text
        # ----------------------------------------------------

        text = content.get_text(
            "\n",
            strip=True,
        )

        text = clean_text(text)

        # ----------------------------------------------------
        # Add page title as contextual information
        # ----------------------------------------------------

        if title:
            title_clean = clean_text(title)

            if title_clean and title_clean not in text:
                text = (
                    f"Page Title: {title_clean}\n\n"
                    + text
                )

        # ----------------------------------------------------
        # Add meta description when useful
        # ----------------------------------------------------

        if description:
            description_clean = clean_text(description)

            if (
                description_clean
                and description_clean not in text
            ):
                text = (
                    f"{text}\n\n"
                    f"Page Description: {description_clean}"
                )

        # ----------------------------------------------------
        # Add footer information
        #
        # Only add it when it contains meaningful content
        # and is not already present in the main content.
        # ----------------------------------------------------

        if footer_text:
            footer_clean = clean_text(footer_text)

            if (
                len(footer_clean.split()) >= 10
                and footer_clean not in text
            ):
                text = (
                    f"{text}\n\n"
                    f"Company Contact Information:\n"
                    f"{footer_clean}"
                )

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------

        text = clean_text(text)

        # ----------------------------------------------------
        # Skip pages with too little useful content
        # ----------------------------------------------------

        if len(text.split()) < 40:
            return None

        # ----------------------------------------------------
        # Return structured page data
        # ----------------------------------------------------

        return {
            "url": normalize_url(response.url),
            "title": title,
            "description": description,
            "headings": headings,
            "content": text,
            "soup": soup,
        }

    except requests.RequestException as e:
        print(
            f"❌ Network error: {url} : {e}"
        )

        return None

    except Exception as e:
        print(
            f"❌ Error scraping {url}: {e}"
        )

        return None


# =========================================================
# CRAWLER
# =========================================================

def crawl(url, depth):
    """
    Crawl internal HTML pages and PDFs up to MAX_DEPTH.

    HTML pages are processed by scrape().
    PDFs are processed by scrape_pdf().

    depth:
        0 = current URL only
        1 = current URL + direct links
        2 = two levels deep
        etc.
    """

    # -----------------------------------------------------
    # MAX PAGE LIMIT
    # -----------------------------------------------------

    if len(visited) >= MAX_PAGES:
        return

    # -----------------------------------------------------
    # NORMALIZE URL
    # -----------------------------------------------------

    url = normalize_url(url)

    # -----------------------------------------------------
    # DEPTH CHECK
    #
    # depth == 0 is valid:
    # it means crawl this page but don't follow children.
    # -----------------------------------------------------

    if depth < 0:
        return

    # -----------------------------------------------------
    # DUPLICATE URL CHECK
    # -----------------------------------------------------

    if url in visited:
        return

    # -----------------------------------------------------
    # URL VALIDATION
    # -----------------------------------------------------

    if not is_valid_url(url):
        return

    # -----------------------------------------------------
    # MARK AS VISITED BEFORE REQUEST
    #
    # This prevents the same URL from being requested
    # repeatedly through different links.
    # -----------------------------------------------------

    visited.add(url)

    print(
        f"🔎 Crawling ({len(visited)}/{MAX_PAGES}) "
        f"[depth={depth}]: {url}"
    )

    # -----------------------------------------------------
    # PDF / DOCUMENT
    # -----------------------------------------------------

    if is_document_url(url):

        try:
            result = scrape_pdf(url)

        except Exception as e:
            print(
                f"❌ Error processing document "
                f"{url}: {e}"
            )
            return

        if result is None:
            return

        pages.append({
            "url": result["url"],
            "title": result.get("title", ""),
            "description": result.get(
                "description",
                "",
            ),
            "headings": result.get(
                "headings",
                {},
            ),
            "content": result["content"],
            "source_type": "pdf",
        })

        print(
            f"📄 PDF indexed: {url}"
        )

        # PDFs normally don't need link crawling
        return

    # -----------------------------------------------------
    # HTML PAGE
    # -----------------------------------------------------

    result = scrape(url)

    if result is None:
        return

    # -----------------------------------------------------
    # SAVE PAGE
    # -----------------------------------------------------

    pages.append({
        "url": result["url"],
        "title": result["title"],
        "description": result["description"],
        "headings": result["headings"],
        "content": result["content"],
        "source_type": "html",
    })

    # -----------------------------------------------------
    # STOP HERE IF MAX DEPTH REACHED
    #
    # We still indexed the current page.
    # We simply don't follow its children.
    # -----------------------------------------------------

    if depth == 0:
        return

    # -----------------------------------------------------
    # EXTRACT INTERNAL LINKS
    # -----------------------------------------------------

    soup = result["soup"]

    for link in soup.find_all("a", href=True):

        href = link.get("href", "").strip()

        if not href:
            continue

        # -------------------------------------------------
        # SKIP NON-WEB LINKS
        # -------------------------------------------------

        if href.startswith((
            "#",
            "mailto:",
            "tel:",
            "javascript:",
            "data:",
        )):
            continue

        # -------------------------------------------------
        # RESOLVE RELATIVE URL
        # -------------------------------------------------

        child = urljoin(
            result["url"],
            href,
        )

        # -------------------------------------------------
        # NORMALIZE CHILD URL
        # -------------------------------------------------

        child = normalize_url(child)

        # -------------------------------------------------
        # VALIDATE CHILD
        # -------------------------------------------------

        if not is_valid_url(child):
            continue

        # -------------------------------------------------
        # CRAWL CHILD
        # -------------------------------------------------

        crawl(
            child,
            depth - 1,
        )

        # -------------------------------------------------
        # RESPECT MAX PAGE LIMIT
        # -------------------------------------------------

        if len(visited) >= MAX_PAGES:
            break

    # -----------------------------------------------------
    # POLITENESS DELAY
    # -----------------------------------------------------

    time.sleep(0.5)

# =========================================================
# PDF SCRAPER
# =========================================================

def scrape_pdf(url):
    """
    Download and extract text from a PDF.

    Returns the same general structure as scrape()
    so the crawler can process HTML and PDF sources
    consistently.
    """

    try:
        print(f"📄 Downloading PDF: {url}")

        response = session.get(
            url,
            headers=HEADERS,
            timeout=30,
        )

        response.raise_for_status()

        # -----------------------------------------------------
        # Validate response
        # -----------------------------------------------------

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        # Some servers incorrectly return application/octet-stream
        # for PDFs, so also check the URL extension.
        is_pdf_content = (
            "application/pdf" in content_type
        )

        is_pdf_url = urlparse(
            url
        ).path.lower().endswith(".pdf")

        if not is_pdf_content and not is_pdf_url:
            print(
                f"⚠️ Not a PDF: {url}"
            )
            return None

        # -----------------------------------------------------
        # Read PDF from memory
        # -----------------------------------------------------

        pdf_bytes = response.content

        if not pdf_bytes:
            print(
                f"⚠️ Empty PDF: {url}"
            )
            return None

        # -----------------------------------------------------
        # Open PDF
        # -----------------------------------------------------

        pdf_file = io.BytesIO(
            pdf_bytes
        )

        reader = PdfReader(
            pdf_file
        )

        # -----------------------------------------------------
        # Extract metadata
        # -----------------------------------------------------

        metadata = reader.metadata or {}

        title = ""

        if metadata.get("/Title"):
            title = str(
                metadata.get("/Title")
            ).strip()

        # -----------------------------------------------------
        # Fallback title from filename
        # -----------------------------------------------------

        if not title:
            filename = Path(
                urlparse(url).path
            ).name

            title = unquote(
                filename
            )

            if title.lower().endswith(".pdf"):
                title = title[:-4]

            # Make filename more readable
            title = re.sub(
                r"[_-]+",
                " ",
                title,
            )

            title = re.sub(
                r"\s+",
                " ",
                title,
            ).strip()

        # -----------------------------------------------------
        # Extract pages
        # -----------------------------------------------------

        page_texts = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:
                text = page.extract_text()

            except Exception as e:
                print(
                    f"⚠️ Could not extract "
                    f"PDF page {page_number}: "
                    f"{e}"
                )
                continue

            if not text:
                continue

            text = clean_text(
                text
            )

            if not text:
                continue

            # Keep page boundaries.
            #
            # This helps later debugging and allows the
            # chatbot to identify where information came from.
            page_texts.append(
                f"[PDF Page {page_number}]\n{text}"
            )

        # -----------------------------------------------------
        # Combine extracted pages
        # -----------------------------------------------------

        text = "\n\n".join(
            page_texts
        )

        text = clean_text(
            text
        )

        # -----------------------------------------------------
        # Validate extracted content
        # -----------------------------------------------------

        if not text:
            print(
                f"⚠️ No text extracted from PDF: "
                f"{url}"
            )

            return None

        if len(text.split()) < 20:
            print(
                f"⚠️ PDF contains too little "
                f"text: {url}"
            )

            return None

        # -----------------------------------------------------
        # Extract simple headings
        #
        # PDF structure is not as reliable as HTML, so we
        # don't pretend every line is an actual heading.
        # -----------------------------------------------------

        headings = {
            "h1": [],
            "h2": [],
            "h3": [],
        }

        # PDF metadata title is useful as the main heading
        if title:
            headings["h1"].append(
                title
            )

        # -----------------------------------------------------
        # Description
        # -----------------------------------------------------

        description = ""

        if metadata.get("/Subject"):
            description = str(
                metadata.get("/Subject")
            ).strip()

        # -----------------------------------------------------
        # Return same structure expected by crawler
        # -----------------------------------------------------

        return {
            "url": normalize_url(
                response.url
            ),
            "title": title,
            "description": description,
            "headings": headings,
            "content": text,
            "source_type": "pdf",
        }

    except requests.RequestException as e:
        print(
            f"❌ PDF network error: "
            f"{url} : {e}"
        )

        return None

    except Exception as e:
        print(
            f"❌ Error processing PDF "
            f"{url}: {e}"
        )

        return None


# =========================================================
# DOCUMENT CREATION
# =========================================================

def build_documents():
    """
    Convert crawled pages/documents into RAG chunks.

    Each chunk keeps:
    - source URL
    - title
    - description
    - heading context
    - source type
    - chunk position
    - word count
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n# ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
        ],
    )

    docs = []

    # -----------------------------------------------------
    # RESET DEDUPLICATION SET
    # -----------------------------------------------------

    seen_chunks.clear()

    # -----------------------------------------------------
    # PROCESS EACH CRAWLED PAGE
    # -----------------------------------------------------

    for page in pages:

        content = page.get("content", "").strip()

        if not content:
            continue

        # -------------------------------------------------
        # Build heading context
        # -------------------------------------------------

        heading_parts = []

        headings = page.get("headings", {})

        for level in ["h1", "h2", "h3"]:

            for heading in headings.get(level, []):

                heading = heading.strip()

                if heading and heading not in heading_parts:
                    heading_parts.append(heading)

        heading_context = " > ".join(
            heading_parts
        )

        # -------------------------------------------------
        # Split page content
        # -------------------------------------------------

        chunks = splitter.split_text(content)

        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):

            chunk = chunk.strip()

            # -------------------------------------------------
            # Minimum length
            # -------------------------------------------------

            if len(chunk) < MIN_CHUNK_LENGTH:
                continue

            # -------------------------------------------------
            # Minimum word count
            # -------------------------------------------------

            word_count = len(chunk.split())

            if word_count < 15:
                continue

            # -------------------------------------------------
            # Add contextual metadata to chunk text
            #
            # This helps the embedding model understand
            # what the chunk is about.
            # -------------------------------------------------

            context_parts = []

            title = page.get(
                "title",
                "",
            ).strip()

            if title:
                context_parts.append(
                    f"Page: {title}"
                )

            if heading_context:
                context_parts.append(
                    f"Sections: {heading_context}"
                )

            if context_parts:

                contextual_chunk = (
                    "\n".join(context_parts)
                    + "\n\n"
                    + chunk
                )

            else:
                contextual_chunk = chunk

            # -------------------------------------------------
            # Deduplicate based on actual content
            #
            # Do NOT include URL/index in this hash.
            # Otherwise identical content on different pages
            # would not be recognized as duplicate content.
            # -------------------------------------------------

            chunk_id = chunk_hash(
                contextual_chunk
            )

            if chunk_id in seen_chunks:
                continue

            seen_chunks.add(chunk_id)

            # -------------------------------------------------
            # Source type
            # -------------------------------------------------

            source_type = page.get(
                "source_type",
                "html",
            )

            # -------------------------------------------------
            # Create document
            # -------------------------------------------------

            docs.append({
                "id": chunk_id,

                "text": contextual_chunk,

                "source": page.get(
                    "url",
                    "",
                ),

                "title": title,

                "description": page.get(
                    "description",
                    "",
                ),

                "headings": headings,

                "heading_context": heading_context,

                "chunk_index": index,

                "total_chunks": total_chunks,

                "word_count": word_count,

                "source_type": source_type,
            })

    print(
        f"📚 Created {len(docs)} RAG chunks "
        f"from {len(pages)} sources"
    )

    return docs



# =========================================================
# EMBEDDINGS
# =========================================================

def build_embeddings(texts):
    """
    Generate embeddings for RAG documents.

    Uses BGE-small through FastEmbed.
    """

    if not texts:
        raise ValueError(
            "No texts provided for embedding."
        )

    print(
        f"🧠 Generating embeddings "
        f"for {len(texts)} chunks..."
    )

    embedder = FastEmbedEmbeddings(
        model_name=EMBED_MODEL
    )

    embeddings = []

    batch_size = 64

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(texts),
        )

        batch = texts[start:end]

        print(
            f"   Embedding "
            f"{start + 1}-{end} "
            f"/ {len(texts)}"
        )

        batch_embeddings = (
            embedder.embed_documents(batch)
        )

        embeddings.extend(
            batch_embeddings
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Validate embedding output
    # -----------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if len(embeddings) != len(texts):
        raise ValueError(
            "Number of embeddings does not "
            "match number of input texts."
        )

    print(
        f"✅ Generated embeddings: "
        f"{embeddings.shape}"
    )

    return embeddings


# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(embeddings):
    """
    Build a FAISS HNSW index using cosine similarity.

    Cosine similarity is implemented as:
        normalized vectors + inner product
    """

    if embeddings is None:
        raise ValueError(
            "Embeddings cannot be None."
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Validate embeddings
    # -----------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got "
            f"shape {embeddings.shape}"
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Cannot build FAISS index "
            "with zero vectors."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or "
            "infinite values."
        )

    # -----------------------------------------------------
    # Normalize vectors
    #
    # Inner product between normalized vectors
    # is equivalent to cosine similarity.
    # -----------------------------------------------------

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    print(
        f"⚡ Building FAISS HNSW index "
        f"({len(embeddings)} vectors, "
        f"dimension={dimension})..."
    )

    # -----------------------------------------------------
    # HNSW parameters
    # -----------------------------------------------------

    M = 32

    index = faiss.IndexHNSWFlat(
        dimension,
        M,
        faiss.METRIC_INNER_PRODUCT,
    )

    # -----------------------------------------------------
    # Build/search quality
    # -----------------------------------------------------

    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64

    # -----------------------------------------------------
    # Add vectors
    # -----------------------------------------------------

    index.add(embeddings)

    print(
        f"✅ Indexed {index.ntotal} vectors"
    )

    return index



# =========================================================
# SAVE FILES
# =========================================================

def save_data(index, docs, embeddings):
    os.makedirs(DATA_DIR, exist_ok=True)

    faiss.write_index(
        index,
        os.path.join(DATA_DIR, "index.faiss")
    )

    np.save(
        os.path.join(DATA_DIR, "embeddings.npy"),
        embeddings
    )

    with open(
        os.path.join(DATA_DIR, "docs.pkl"),
        "wb"
    ) as f:
        pickle.dump(docs, f)

    print("✅ Saved:")
    print("   - index.faiss")
    print("   - embeddings.npy")
    print("   - docs.pkl")

# =========================================================
# MAIN INDEX BUILD
# =========================================================

def main():
    """
    Build the complete Company RAG index.

    Pipeline:
        Website
          ↓
        Crawl
          ↓
        HTML / PDF extraction
          ↓
        Cleaning
          ↓
        Chunking
          ↓
        Embeddings
          ↓
        FAISS
          ↓
        Save
    """

    # -----------------------------------------------------
    # Reset global state
    #
    # Important when running main() more than once in the
    # same Python process/notebook.
    # -----------------------------------------------------

    visited.clear()
    pages.clear()
    seen_chunks.clear()

    print("\n" + "=" * 60)
    print("🚀 NOVATECH ROBO RAG INDEX BUILD")
    print("=" * 60)

    print("\n⚙️ Configuration")
    print(f"   Base URL   : {BASE_URL}")
    print(f"   Max depth  : {MAX_DEPTH}")
    print(f"   Max pages  : {MAX_PAGES}")
    print(f"   Chunk size : {CHUNK_SIZE}")
    print(f"   Overlap    : {CHUNK_OVERLAP}")
    print(f"   Embedding  : {EMBED_MODEL}")

    # -----------------------------------------------------
    # 1. CRAWL
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("🌐 STEP 1 — CRAWLING WEBSITE")
    print("-" * 60)

    crawl(
        BASE_URL,
        MAX_DEPTH,
    )

    print(
        f"\n✅ URLs visited: {len(visited)}"
    )

    print(
        f"✅ Sources extracted: {len(pages)}"
    )

    # -----------------------------------------------------
    # Source statistics
    # -----------------------------------------------------

    html_pages = sum(
        1
        for page in pages
        if page.get("source_type") == "html"
    )

    pdf_pages = sum(
        1
        for page in pages
        if page.get("source_type") == "pdf"
    )

    print(
        f"   HTML pages : {html_pages}"
    )

    print(
        f"   PDF files  : {pdf_pages}"
    )

    # -----------------------------------------------------
    # 2. BUILD DOCUMENT CHUNKS
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("📚 STEP 2 — BUILDING DOCUMENT CHUNKS")
    print("-" * 60)

    docs = build_documents()

    print(
        f"\n✅ Final chunks: {len(docs)}"
    )

    if not docs:
        print(
            "\n❌ No documents were created."
        )

        print(
            "   Check crawling, extraction, "
            "and chunking."
        )

        return

    # -----------------------------------------------------
    # Chunk statistics
    # -----------------------------------------------------

    html_chunks = sum(
        1
        for doc in docs
        if doc.get("source_type") == "html"
    )

    pdf_chunks = sum(
        1
        for doc in docs
        if doc.get("source_type") == "pdf"
    )

    print(
        f"   HTML chunks : {html_chunks}"
    )

    print(
        f"   PDF chunks  : {pdf_chunks}"
    )

    # -----------------------------------------------------
    # 3. PREPARE TEXT
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("📝 STEP 3 — PREPARING TEXT FOR EMBEDDINGS")
    print("-" * 60)

    texts = [
        doc["text"]
        for doc in docs
    ]

    if len(texts) != len(docs):
        raise RuntimeError(
            "Document/text count mismatch."
        )

    print(
        f"✅ Texts prepared: {len(texts)}"
    )

    # -----------------------------------------------------
    # 4. GENERATE EMBEDDINGS
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("🧠 STEP 4 — GENERATING EMBEDDINGS")
    print("-" * 60)

    embeddings = build_embeddings(
        texts
    )

    print(
        f"✅ Embeddings shape: "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------
    # Verify alignment
    # -----------------------------------------------------

    if len(embeddings) != len(docs):
        raise RuntimeError(
            "CRITICAL: Number of embeddings "
            "does not match number of documents."
        )

    # -----------------------------------------------------
    # 5. BUILD FAISS INDEX
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("⚡ STEP 5 — BUILDING FAISS INDEX")
    print("-" * 60)

    index = build_faiss_index(
        embeddings
    )

    if index.ntotal != len(docs):
        raise RuntimeError(
            "CRITICAL: FAISS vector count "
            "does not match document count."
        )

    print(
        f"✅ FAISS vectors: {index.ntotal}"
    )

    # -----------------------------------------------------
    # 6. SAVE EVERYTHING
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("💾 STEP 6 — SAVING INDEX")
    print("-" * 60)

    save_data(
        index,
        docs,
        embeddings,
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("🎉 INDEX BUILD COMPLETE")
    print("=" * 60)

    print(
        f"🌐 URLs visited : {len(visited)}"
    )

    print(
        f"📄 HTML sources : {html_pages}"
    )

    print(
        f"📕 PDF sources  : {pdf_pages}"
    )

    print(
        f"🧩 RAG chunks   : {len(docs)}"
    )

    print(
        f"🧠 Embeddings   : {len(embeddings)}"
    )

    print(
        f"⚡ FAISS vectors: {index.ntotal}"
    )

    print("=" * 60)



# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()