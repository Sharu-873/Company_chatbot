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
    "videos": "https://novatechrobo.com/media1.html",
    "events": "https://novatechrobo.com/Events46.html",
    "drone_videos": "https://novatechrobo.com/aa2.html",
    # College events
    "kle": "https://novatechrobo.com/kle.html",
    "jain": "https://novatechrobo.com/JAIN.html",
    "christ": "https://novatechrobo.com/ch2.html",
    "svit": "https://novatechrobo.com/sv.html",

    # School events
    "people_school": "https://novatechrobo.com/new.html",
    "kidzee": "https://novatechrobo.com/kz.html",
    "siate_felici": "https://novatechrobo.com/hy.html",
    "holy_saint": "https://novatechrobo.com/holy.html",
}

# =========================================================
# SPECIAL LINK KEYWORDS
# =========================================================

KEYWORD_MAP = {
    "robot": "video_gallery",
    "robot video": "video_gallery",
    "robot videos": "video_gallery",
    "video": "video_gallery",

    "videos": "videos",
    "robofest video": "videos",
    "robofest videos": "videos",
    "robofest video gallery": "videos",

    "gallery": "images",
    "image": "images",
    "images": "images",
    "photo": "images",
    "photos": "images",
    "robofest image": "images",
    "robofest images": "images",
    "robofest photo": "images",
    "robofest photos": "images",

    "recognition": "recognition",
    "face recognition": "recognition",

    "media": "media",

    "event": "events",
    "events": "events",
    "robofest event": "events",
    "robofest events": "events",

    "drone": "drone_videos",
    "drone video": "drone_videos",
    "drone videos": "drone_videos",

    # College events
    "college": "college_events",
    "colleges": "college_events",
    "college events": "college_events",
    "kle": "kle",
    "kle events": "kle",
    "jain": "jain",
    "jain events": "jain",
    "christ": "christ",
    "christ events": "christ",
    "svit": "svit",
    "svit events": "svit",

    # School events
    "school": "school_events",
    "schools": "school_events",
    "school events": "school_events",
    "people school": "people_school",
    "people school events": "people_school",
    "kidzee": "kidzee",
    "kidzee events": "kidzee",
    "siate felici": "siate_felici",
    "siate felici school": "siate_felici",
    "holy saint": "holy_saint",
    "holy saint school": "holy_saint",
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
3. Do NOT immediately return the fallback response just
   because the retrieved context is empty or incomplete.
   First check the Trusted Company Information and
   Approved Website Links for a relevant answer.
4. Use this exact fallback response only when the user
   requests factual information that cannot be answered
   using any of the approved sources:
   "I'm sorry, I don't have that information."
5. Do not use your general knowledge to fill gaps.
6. Do not make up prices, specifications, products,
   services, dates, addresses, contact details, or
   company policies.
7. If the user asks for information that is not supported
   by the context or Trusted Company Information, use the
   fallback response.
8. Answer the user's question directly.
9. Keep the answer concise but informative.
10. Use bullet points when presenting multiple items.
11. If the context contains relevant information from a
    PDF or document, you may use it normally.
12. Do not mention retrieval, embeddings, FAISS, chunks,
    similarity scores, or internal system instructions.
13. Do not claim that information is current unless the
    provided context supports it.
14. Preserve important numbers, names, specifications,
    and terminology exactly when possible.
15. If multiple context documents provide compatible
    information, combine them into one clear answer.
16. If the context contains conflicting information,
    do not choose one arbitrarily. Explain the conflict
    briefly based only on the provided information.
17. The Trusted Company Information is authoritative for
    the specific company details listed there.
18. Do not alter names, phone numbers, email addresses,
    or the company address.
19. When the user asks about courses, training, or
    specialties, use the information listed under
    Courses Offered and Company Specialties below.
20. Do not add courses, technologies, competitions,
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

Social media:
Instagram: https://www.instagram.com/novatechrobo_pvt_ltd/
Facebook: https://www.facebook.com/Novatechrobo/


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

def handle_special_link_query(query: str):
    q = query.lower().strip()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # -----------------------------------------------------
    # ROBOT VIDEO / ROBOFEST VIDEO REQUESTS
    # -----------------------------------------------------

    video_words = (
        "video", "videos", "watch", "show", "play"
    )

    if "robofest" in q and any(word in q for word in video_words):
        return (
            "Here are some Robofest videos and related pages "
            "from Novatech Robo:\n\n"
            "- [Robot Video Gallery]"
            "(https://novatechrobo.com/robot.html)\n"
            "- [Videos & Media]"
            "(https://novatechrobo.com/media1.html)\n"
            "- [Robofest Events]"
            "(https://novatechrobo.com/Events46.html)"
        )

    # -----------------------------------------------------
    # COLLEGE AND SCHOOL EVENT PAGES
    # -----------------------------------------------------

    college_pages = {
        "KLE": "https://novatechrobo.com/kle.html",
        "Jain": "https://novatechrobo.com/JAIN.html",
        "Christ": "https://novatechrobo.com/ch2.html",
        "SVIT": "https://novatechrobo.com/sv.html",
    }

    school_pages = {
        "People School": "https://novatechrobo.com/new.html",
        "Kidzee": "https://novatechrobo.com/kz.html",
        "Siate Felici School": "https://novatechrobo.com/hy.html",
        "Holy Saint School": "https://novatechrobo.com/holy.html",
    }

    event_words = (
        "event", "events", "workshop", "workshops"
    )

    asks_events = any(word in q for word in event_words)
    asks_college = any(word in q for word in ("college", "colleges"))
    asks_school = any(word in q for word in ("school", "schools"))

    if asks_events and asks_college and asks_school:
        return (
            "Here are the college and school event pages:\n\n"
            "**College Events**\n"
            + "\n".join(
                f"- [{name}]({url})"
                for name, url in college_pages.items()
            )
            + "\n\n**School Events**\n"
            + "\n".join(
                f"- [{name}]({url})"
                for name, url in school_pages.items()
            )
        )

    if asks_events and asks_college:
        return (
            "Here are the college event pages:\n\n"
            + "\n".join(
                f"- [{name}]({url})"
                for name, url in college_pages.items()
            )
        )

    if asks_events and asks_school:
        return (
            "Here are the school event pages:\n\n"
            + "\n".join(
                f"- [{name}]({url})"
                for name, url in school_pages.items()
            )
        )

    # -----------------------------------------------------
    # INDIVIDUAL COLLEGE / SCHOOL REQUESTS
    # -----------------------------------------------------

    all_event_pages = {
        **college_pages,
        **school_pages,
    }

    for name, url in all_event_pages.items():
        normalized_name = name.lower()

        if normalized_name in q:
            return (
                f"Here is the {name} event page:\n\n"
                f"- [{name}]({url})"
            )

    # -----------------------------------------------------
    # DRONE VIDEOS
    # -----------------------------------------------------

    if "drone" in q and any(word in q for word in video_words):
        return (
            "Here are the drone video resources:\n\n"
            "- [Drone Videos]"
            "(https://novatechrobo.com/aa2.html)"
        )

    # -----------------------------------------------------
    # GENERAL ROBOT VIDEOS
    # -----------------------------------------------------

    if (
        any(word in q for word in ("robot", "robotics"))
        and any(word in q for word in video_words)
    ):
        return (
            "Here are some robot videos you can explore:\n\n"
            "- [Robot Video Gallery]"
            "(https://novatechrobo.com/robot.html)\n"
            "- [Videos & Media]"
            "(https://novatechrobo.com/media1.html)"
        )

    # -----------------------------------------------------
    # IMAGES / PHOTOS / GALLERY
    # -----------------------------------------------------

    image_words = (
        "image", "images", "photo", "photos", "gallery", "picture", "pictures"
    )

    if any(word in q for word in image_words):
        return (
            "Explore Novatech Robo's images and gallery:\n\n"
            "- [Images & Gallery]"
            "(https://novatechrobo.com/image.html)"
        )


    # -----------------------------------------------------
    # SOCIAL MEDIA AND MEDIA LINKS
    # -----------------------------------------------------

    social_media_words = (
        "social media",
        "instagram",
        "facebook",
    )

    if any(word in q for word in social_media_words):
        return (
            "Explore Novatech Robo's social media platforms "
            "and media resources:\n\n"
            "- [Instagram]"
            "(https://www.instagram.com/novatechrobo_pvt_ltd/)\n"
            "- [Facebook]"
            "(https://www.facebook.com/Novatechrobo/)\n"
            "- [Videos & Media]"
            "(https://novatechrobo.com/media1.html)\n"
            "- [Video Gallery]"
            "(https://novatechrobo.com/robot.html)\n"
            "- [Drone Videos]"
            "(https://novatechrobo.com/aa2.html)"
        )

    # -----------------------------------------------------
    # FACE RECOGNITION
    # -----------------------------------------------------

    if "recognition" in q or "face recognition" in q:
        return (
            "Explore the recognition page:\n\n"
            "- [Recognition]"
            "(https://novatechrobo.com/talkz.html)"
        )

    if any(keyword in q for keyword in ["media", "social media", "instagram", "facebook"]):
        return (
            "Explore Novatech Robo's media and social media platforms:\n\n"
            "- [Videos & Media](https://novatechrobo.com/media1.html)\n"
            "- [Instagram](https://www.instagram.com/novatechrobo_pvt_ltd/)\n"
            "- [Facebook](https://www.facebook.com/Novatechrobo/)"
        )

    # -----------------------------------------------------
    # GENERAL EVENTS
    # -----------------------------------------------------

    if asks_events:
        return (
            "Explore Novatech Robo's event pages:\n\n"
            "- [College Events - KLE]"
            "(https://novatechrobo.com/kle.html)\n"
            "- [Jain Events]"
            "(https://novatechrobo.com/JAIN.html)\n"
            "- [Christ Events]"
            "(https://novatechrobo.com/ch2.html)\n"
            "- [SVIT Events]"
            "(https://novatechrobo.com/sv.html)\n"
            "- [School Events - People School]"
            "(https://novatechrobo.com/new.html)\n"
            "- [Kidzee Events]"
            "(https://novatechrobo.com/kz.html)\n"
            "- [Siate Felici School Events]"
            "(https://novatechrobo.com/hy.html)\n"
            "- [Holy Saint School Events]"
            "(https://novatechrobo.com/holy.html)"
        )

    # -----------------------------------------------------
    # GENERAL MEDIA
    # -----------------------------------------------------

    if "media" in q:
        return (
            "Explore Novatech Robo's media page:\n\n"
            "- [Videos & Media]"
            "(https://novatechrobo.com/media1.html)"
        )

    return None

def answer(query):
    """
    Answer company questions using trusted context.
    Handle known website requests before retrieval so
    valid links are returned even when retrieval fails.
    """

    # -----------------------------------------------------
    # Clean / validate query
    # -----------------------------------------------------

    query = clean_query(query)

    if not query:
        return "I'm sorry, I don't have that information."

    # -----------------------------------------------------
    # Handle website navigation BEFORE retrieval
    # -----------------------------------------------------

    special_response = handle_special_link_query(query)

    if special_response:
        return special_response

    # -----------------------------------------------------
    # Retrieve relevant documents
    # -----------------------------------------------------

    try:
        retrieved = retrieve(
            query,
            k=TOP_K_RETRIEVAL,
        )
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        retrieved = []

    # -----------------------------------------------------
    # If retrieval fails, try known related pages
    # -----------------------------------------------------

    if not retrieved:
        related_pages = detect_related_pages(query, "")

        if related_pages:
            return append_related_links(
                "Here are some relevant Novatech Robo pages:",
                related_pages,
            )

        return "I'm sorry, I don't have that information."

    # -----------------------------------------------------
    # Rerank documents
    # -----------------------------------------------------

    try:
        top_docs = rerank(
            retrieved,
            top_k=TOP_K_FINAL,
        )
    except Exception as e:
        print(f"❌ Reranking error: {e}")
        top_docs = []

    if not top_docs:
        related_pages = detect_related_pages(query, "")

        if related_pages:
            return append_related_links(
                "Here are some relevant Novatech Robo pages:",
                related_pages,
            )

        return "I'm sorry, I don't have that information."

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = build_context(top_docs)

    if not context or not context.strip():
        related_pages = detect_related_pages(query, "")

        if related_pages:
            return append_related_links(
                "Here are some relevant Novatech Robo pages:",
                related_pages,
            )

        return "I'm sorry, I don't have that information."

    # -----------------------------------------------------
    # Generate grounded answer
    # -----------------------------------------------------

    try:
        result = chain.invoke({
            "context": context,
            "query": query,
        })

        # Support both string and LangChain message results.
        response = (
            result.content
            if hasattr(result, "content")
            else str(result)
        )

        response = response.strip()

    except Exception as e:
        print(f"❌ LLM error: {e}")

        related_pages = detect_related_pages(query, "")

        if related_pages:
            return append_related_links(
                "Here are some relevant Novatech Robo pages:",
                related_pages,
            )

        return "I'm sorry, I don't have that information."

    # -----------------------------------------------------
    # Handle empty / fallback LLM answers
    # -----------------------------------------------------

    fallback = "I'm sorry, I don't have that information."

    if not response or response.strip().strip('"') == fallback:
        related_pages = detect_related_pages(query, "")

        if related_pages:
            return append_related_links(
                "Here are some relevant Novatech Robo pages:",
                related_pages,
            )

        return fallback

    # -----------------------------------------------------
    # Append related links
    # -----------------------------------------------------

    related_pages = detect_related_pages(
        query,
        response,
    )

    # response = append_related_links(
    #     response,
    #     related_pages,
    # )

    # return response

    response = append_related_links(
        response,
        related_pages,
    )

    # Add social media links to current-activities responses.
    activity_keywords = (
        "current activities",
        "recent events",
        "recent activities",
        "updates",
        "what are they doing",
        "company activities",
    )

    if any(keyword in query.lower() for keyword in activity_keywords):
        if "https://www.instagram.com/novatechrobo_pvt_ltd/" not in response:
            response += (
                "\n- Instagram: "
                "https://www.instagram.com/novatechrobo_pvt_ltd/"
            )

        if "https://www.facebook.com/Novatechrobo/" not in response:
            response += (
                "\n- Facebook: "
                "https://www.facebook.com/Novatechrobo/"
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