"""
Document ingestion: fetches article summaries from Wikipedia and dumps
them into the local ChromaDB collection.

Run from the project root:
    python app/ingest.py
"""

import sys
import requests

WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# (chroma_id, Wikipedia article title)
CLINICAL_TOPICS: list[tuple[str, str]] = [
    ("clinical_trial", "Clinical trial"),
    ("randomized_controlled_trial", "Randomized controlled trial"),
    ("pharmacokinetics", "Pharmacokinetics"),
    ("drug_metabolism", "Drug metabolism"),
    ("adverse_drug_reaction", "Adverse drug reaction"),
    ("clinical_endpoint", "Clinical endpoint"),
    ("placebo", "Placebo"),
    ("blinded_experiment", "Blinded experiment"),
    ("informed_consent", "Informed consent"),
    ("biomarker_medicine", "Biomarker (medicine)"),
]


def fetch_wikipedia_summary(title: str) -> str:
    """Return the plain-text extract for a Wikipedia article."""
    url = WIKIPEDIA_REST.format(requests.utils.quote(title))
    headers = {"User-Agent": "fastapi-tutorial-ingest/1.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("extract", "")


def run_ingestion() -> int:
    """Fetch all topics and upsert into ChromaDB. Returns the number ingested."""
    # Late import so the module-level ChromaDB client in sample.py is only
    # initialised when this function is actually called.
    from sample import ingest_documents

    ids: list[str] = []
    documents: list[str] = []

    for doc_id, title in CLINICAL_TOPICS:
        print(f"  Fetching '{title}' ...", end=" ", flush=True)
        try:
            text = fetch_wikipedia_summary(title)
            if text:
                ids.append(doc_id)
                documents.append(text)
                print(f"OK  ({len(text):,} chars)")
            else:
                print("SKIP (empty extract)")
        except requests.RequestException as exc:
            print(f"ERROR — {exc}", file=sys.stderr)

    if documents:
        ingest_documents(documents=documents, ids=ids)
        print(f"\nIngested {len(documents)} document(s) into ChromaDB.")
    else:
        print("No documents were ingested.")

    return len(documents)


if __name__ == "__main__":
    run_ingestion()
