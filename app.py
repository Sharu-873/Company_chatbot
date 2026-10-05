import re
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import faiss
import pickle
from langchain_core.documents import Document
import hashlib

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================================================
# CONFIG
# =========================================================

VECTOR_DB_PATH = "data"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Number of candidates retrieved from FAISS
TOP_K_RETRIEVAL = 8

# Number of chunks finally passed to the LLM
TOP_K_FINAL = 4

# Minimum similarity required for a result to be considered
# relevant.
#
# This should be treated as a filtering signal, not as a
# guarantee that every relevant question will score above it.
MIN_SIMILARITY = 0.45


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)


# =========================================================
# SPECIAL PAGE LINKS
# =========================================================

PAGE_MAP = {
    "video_gallery": "https://novatechrobo.com/robot.html",
    "images": "https://novatechrobo.com/image.html",
    "recognition": "https://novatechrobo.com/talkz.html",
    "media": "https://novatechrobo.com/media1.html",
    "events": "https://novatechrobo.com/Events46.html",
    "drone_videos": "https://novatechrobo.com/aa2.html",
}


# =========================================================
# SPECIAL LINK KEYWORDS
# =========================================================

KEYWORD_MAP = {
    "robot": "video_gallery",
    "video": "video_gallery",
    "gallery": "images",
    "image": "images",
    "images": "images",
    "photo": "images",
    "photos": "images",
    "recognition": "recognition",
    "face recognition": "recognition",
    "media": "media",
    "event": "events",
    "events": "events",
    "drone": "drone_videos",
}


# =========================================================
# EMBEDDINGS
# =========================================================

print("🧠 Loading embedding model...")

embedder = FastEmbedEmbeddings(
    model_name=EMBED_MODEL
)

print("✅ Embedding model loaded")


# =========================================================
# LOAD FAISS INDEX
# =========================================================

INDEX_PATH = os.path.join(
    VECTOR_DB_PATH,
    "index.faiss",
)

print(
    f"⚡ Loading FAISS index: {INDEX_PATH}"
)

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"✅ FAISS vectors loaded: "
    f"{index.ntotal}"
)


# =========================================================
# LOAD DOCUMENT METADATA
# =========================================================

DOCS_PATH = os.path.join(
    VECTOR_DB_PATH,
    "docs.pkl",
)

print(
    f"📚 Loading document metadata: "
    f"{DOCS_PATH}"
)

with open(
    DOCS_PATH,
    "rb",
) as f:
    stored_docs = pickle.load(f)

print(
    f"✅ Documents loaded: "
    f"{len(stored_docs)}"
)


# =========================================================
# CRITICAL INDEX/DOCUMENT ALIGNMENT CHECK
# =========================================================

if index.ntotal != len(stored_docs):
    raise RuntimeError(
        "CRITICAL: FAISS index count does not "
        "match stored document count.\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Documents: {len(stored_docs)}"
    )


# =========================================================
# CONVERT TO LANGCHAIN DOCUMENTS
# =========================================================

documents = []

for d in stored_docs:

    documents.append(
        Document(
            page_content=d.get(
                "text",
                "",
            ),

            metadata={
                # -----------------------------------------
                # Source
                # -----------------------------------------

                "source": d.get(
                    "source",
                    "",
                ),

                "title": d.get(
                    "title",
                    "",
                ),

                # -----------------------------------------
                # Additional metadata
                # -----------------------------------------

                "description": d.get(
                    "description",
                    "",
                ),

                "source_type": d.get(
                    "source_type",
                    "html",
                ),

                "heading_context": d.get(
                    "heading_context",
                    "",
                ),

                "headings": d.get(
                    "headings",
                    {},
                ),

                "chunk_index": d.get(
                    "chunk_index",
                    0,
                ),

                "total_chunks": d.get(
                    "total_chunks",
                    1,
                ),

                "word_count": d.get(
                    "word_count",
                    len(
                        d.get(
                            "text",
                            "",
                        ).split()
                    ),
                ),
            },
        )
    )


print(
    f"✅ LangChain documents ready: "
    f"{len(documents)}"
)

# =========================================================
# QUERY CLEANING
# =========================================================

def clean_query(query):
    """
    Normalize the user's query before embedding.
    """

    if query is None:
        return ""

    query = str(query).strip()

    # Normalize whitespace
    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    return query

def chunk_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    query,
    k=TOP_K_RETRIEVAL,
    min_similarity=MIN_SIMILARITY,
):
    """
    Retrieve the most relevant documents from FAISS.

    Returns results ordered by descending similarity.
    """

    # -----------------------------------------------------
    # Clean query
    # -----------------------------------------------------

    query = clean_query(query)

    if not query:
        return []

    # -----------------------------------------------------
    # Determine search size
    # -----------------------------------------------------

    if index.ntotal == 0:
        return []

    search_k = min(
        max(k, 1),
        index.ntotal,
    )

    # -----------------------------------------------------
    # Embed query
    # -----------------------------------------------------

    query_embedding = embedder.embed_query(
        query
    )

    query_embedding = np.asarray(
        [query_embedding],
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Validate query embedding
    # -----------------------------------------------------

    if query_embedding.ndim != 2:
        raise ValueError(
            "Invalid query embedding shape: "
            f"{query_embedding.shape}"
        )

    if not np.isfinite(
        query_embedding
    ).all():
        raise ValueError(
            "Query embedding contains "
            "NaN or infinite values."
        )

    # -----------------------------------------------------
    # Normalize for cosine similarity
    #
    # The FAISS index uses inner product.
    # Both stored vectors and query vectors must
    # therefore be normalized.
    # -----------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )

    # -----------------------------------------------------
    # FAISS SEARCH
    # -----------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        search_k,
    )

    # -----------------------------------------------------
    # Build retrieval results
    # -----------------------------------------------------

    results = []

    seen_text = set()
    seen_sources = set()

    for rank, (
        score,
        idx,
    ) in enumerate(
        zip(
            scores[0],
            indices[0],
        ),
        start=1,
    ):

        # -------------------------------------------------
        # Invalid FAISS result
        # -------------------------------------------------

        if idx < 0:
            continue

        if idx >= len(documents):
            continue

        score = float(score)

        # -------------------------------------------------
        # Similarity filtering
        # -------------------------------------------------

        if score < min_similarity:
            continue

        doc = documents[idx]

        text = doc.page_content.strip()

        if not text:
            continue

        # -------------------------------------------------
        # Remove exact duplicate chunks
        # -------------------------------------------------

        text_hash = chunk_hash(
            text
        )

        if text_hash in seen_text:
            continue

        seen_text.add(
            text_hash
        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        metadata = dict(
            doc.metadata
        )

        source = metadata.get(
            "source",
            "",
        )

        source_type = metadata.get(
            "source_type",
            "html",
        )

        # -------------------------------------------------
        # Add retrieval information
        # -------------------------------------------------

        metadata["retrieval_rank"] = rank

        metadata["similarity"] = score

        metadata["source_type"] = (
            source_type
        )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append({
            "text": text,

            "metadata": metadata,

            "score": score,

            "index": int(idx),

            "rank": rank,

        })

    # -----------------------------------------------------
    # Final ordering
    # -----------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


# =========================================================
# RESULT DEDUPLICATION
# =========================================================

def deduplicate_results(results):
    """
    Remove duplicate retrieved chunks.

    Uses the full normalized text rather than only the
    first 120 characters.
    """

    seen_hashes = set()

    unique = []

    for result in results:

        text = result.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        # -------------------------------------------------
        # Normalize whitespace before hashing
        # -------------------------------------------------

        normalized_text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip().lower()

        text_id = chunk_hash(
            normalized_text
        )

        if text_id in seen_hashes:
            continue

        seen_hashes.add(
            text_id
        )

        unique.append(
            result
        )

    return unique


# =========================================================
# RERANK
# =========================================================

def rerank(
    results,
    top_k=TOP_K_FINAL,
):
    """
    Rank retrieved results by similarity and return
    the strongest unique results.

    Higher cosine similarity = better match.
    """

    if not results:
        return []

    # -----------------------------------------------------
    # Remove duplicate chunks
    # -----------------------------------------------------

    unique_results = deduplicate_results(
        results
    )

    # -----------------------------------------------------
    # Sort by similarity
    # -----------------------------------------------------

    ranked = sorted(
        unique_results,
        key=lambda x: float(
            x.get("score", 0.0)
        ),
        reverse=True,
    )

    # -----------------------------------------------------
    # Limit final context
    # -----------------------------------------------------

    top_k = max(
        int(top_k),
        1,
    )

    return ranked[:top_k]


# =========================================================
# RELATED PAGE DETECTION
# =========================================================

def detect_related_pages(
    query,
    response="",
):
    """
    Detect website pages that may be useful to the user.

    The user's query gets priority. The generated response
    is used only as a secondary signal.
    """

    query_text = clean_query(
        query
    )

    response_text = clean_query(
        response
    )

    found = []

    # -----------------------------------------------------
    # 1. Check user query first
    # -----------------------------------------------------

    for keyword, page_key in KEYWORD_MAP.items():

        keyword = keyword.lower().strip()

        if not keyword:
            continue

        if keyword in query_text:
            if page_key not in found:
                found.append(page_key)

    # -----------------------------------------------------
    # 2. Check response only if needed
    #
    # This helps discover a relevant page mentioned by the
    # answer, but prevents response text from becoming the
    # primary driver.
    # -----------------------------------------------------

    if not found:

        for keyword, page_key in KEYWORD_MAP.items():

            keyword = keyword.lower().strip()

            if not keyword:
                continue

            if keyword in response_text:

                if page_key not in found:
                    found.append(page_key)

    return found


# =========================================================
# APPEND RELATED LINKS
# =========================================================

def append_related_links(
    response,
    page_keys,
):
    """
    Append useful internal website links to the answer.
    """

    if not response:
        response = ""

    if not page_keys:
        return response

    links = []

    for key in page_keys:

        url = PAGE_MAP.get(
            key
        )

        if not url:
            continue

        label = (
            key
            .replace("_", " ")
            .title()
        )

        links.append(
            f"- {label}: {url}"
        )

    if not links:
        return response

    # -----------------------------------------------------
    # Avoid duplicate section
    # -----------------------------------------------------

    if "Related Links:" in response:
        return response

    response = (
        response.rstrip()
        + "\n\n"
        + "Related Links:\n"
        + "\n".join(links)
    )

    return response

# =========================================================
# BUILD CONTEXT
# =========================================================
# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(docs):
    """
    Build the context passed to the LLM.

    Each retrieved chunk includes source metadata so the
    model can understand where the information came from.
    """

    if not docs:
        return ""

    context_parts = []

    for i, doc in enumerate(
        docs,
        start=1,
    ):

        metadata = doc.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "",
        )

        title = metadata.get(
            "title",
            "",
        )

        source_type = metadata.get(
            "source_type",
            "html",
        )

        heading_context = metadata.get(
            "heading_context",
            "",
        )

        similarity = metadata.get(
            "similarity",
            doc.get("score", 0.0),
        )

        content = doc.get(
            "text",
            "",
        ).strip()

        if not content:
            continue

        # -------------------------------------------------
        # Human-readable source label
        # -------------------------------------------------

        if source_type == "pdf":
            source_label = "PDF"
        else:
            source_label = "Website"

        context_parts.append(
            f"""
Document {i}
Source type: {source_label}
Title: {title}
URL: {source}
Section: {heading_context}
Relevance score: {similarity:.3f}

Content:
{content}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )

# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant for Novatech Robo.

Your job is to answer the user's question using ONLY
information supported by the provided context or the
Trusted Company Information below.

IMPORTANT RULES:

1. Use only the supplied context and Trusted Company Information.
2. Do not invent, guess, or assume company facts.
3. If the available information does not contain enough
   information to answer the question, respond exactly with:
   "I'm sorry, I don't have that information."
4. Do not use your general knowledge to fill gaps.
5. Do not make up prices, specifications, products,
   services, dates, addresses, contact details, or
   company policies.
6. If the user asks for information that is not supported
   by the context or Trusted Company Information, use the
   fallback response.
7. Answer the user's question directly.
8. Keep the answer concise but informative.
9. Use bullet points when presenting multiple items.
10. If the context contains relevant information from a
    PDF or document, you may use it normally.
11. Do not mention retrieval, embeddings, FAISS, chunks,
    similarity scores, or internal system instructions.
12. Do not claim that information is current unless the
    provided context supports it.
13. Preserve important numbers, names, specifications,
    and terminology exactly when possible.
14. If multiple context documents provide compatible
    information, combine them into one clear answer.
15. If the context contains conflicting information,
    do not choose one arbitrarily. Explain the conflict
    briefly based only on the provided information.
16. The Trusted Company Information is authoritative for
    the specific company details listed there.
17. Do not alter names, phone numbers, email addresses,
    or the company address.
18. When the user asks about courses, training, or
    specialties, use the information listed under
    Courses Offered and Company Specialties below.
19. Do not add courses, technologies, competitions,
    products, or specialties that are not explicitly
    listed in the trusted information or supplied context.

TRUSTED COMPANY INFORMATION:

Company:
Novatech Robo Pvt Ltd.

CEO:
Mr. I A Khan

Address:
Novatechrobo
#40, First Floor, 36th Cross,
4th T Block, Jayanagar,
Bengaluru - 560041
Above DRY FRUIT HOUSE

Contact:
+91 9019247247
+91 9341253057
+91 6366618044

Email:
robotic999@gmail.com


COURSES OFFERED:

- STEM (Robotics)
- Drone
- 3D Printing
- Raspberry Pi
- Arduino


COMPANY SPECIALTIES:

- STEM and Robotics Training
- College workshops
- Robofest Competition
- Humanoid Robots


SOURCE HANDLING:

- Website content and PDF content are valid sources.
- The Trusted Company Information above is also a valid source.
- The source metadata is provided to help interpret the
  retrieved content.
- Do not invent information that is not present in either
  the Trusted Company Information or the supplied context.


Context:
{context}

Question:
{query}

Answer:
"""
)




chain = (
    prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# ANSWER
# =========================================================

def answer(query):
    """
    Retrieve relevant company information and generate
    a grounded answer.
    """

    # -----------------------------------------------------
    # Clean/validate query
    # -----------------------------------------------------

    query = clean_query(
        query
    )

    if not query:
        return (
            "I'm sorry, I don't have that information."
        )

    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    retrieved = retrieve(
        query,
        k=TOP_K_RETRIEVAL,
    )

    if not retrieved:
        return (
            "I'm sorry, I don't have that information."
        )

    # -----------------------------------------------------
    # Rerank / select final context
    # -----------------------------------------------------

    top_docs = rerank(
        retrieved,
        top_k=TOP_K_FINAL,
    )

    if not top_docs:
        return (
            "I'm sorry, I don't have that information."
        )

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = build_context(
        top_docs
    )

    if not context.strip():
        return (
            "I'm sorry, I don't have that information."
        )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    try:

        response = chain.invoke({
            "context": context,
            "query": query,
        })

        response = (
            str(response)
            .strip()
        )

    except Exception as e:

        print(
            f"❌ LLM error: {e}"
        )

        return (
            "I'm sorry, I don't have that information."
        )

    # -----------------------------------------------------
    # Empty response protection
    # -----------------------------------------------------

    if not response:
        return (
            "I'm sorry, I don't have that information."
        )

    # -----------------------------------------------------
    # Detect useful related pages
    # -----------------------------------------------------

    related_pages = detect_related_pages(
        query,
        response,
    )

    # -----------------------------------------------------
    # Append related links
    # -----------------------------------------------------

    response = append_related_links(
        response,
        related_pages,
    )

    return response


# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":

    while True:

        query = input("\nQuestion: ")

        if query.lower() in ["exit", "quit"]:
            break

        response = answer(query)

        print("\nAnswer:")
        print(response)