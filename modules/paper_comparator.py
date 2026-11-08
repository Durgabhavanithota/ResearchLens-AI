
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_text(text):
    """Clean research paper text for comparison."""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.strip()


def compare_papers(papers):
    """
    Compare uploaded research papers using TF-IDF
    and cosine similarity.
    
    papers: list of dictionaries containing name and text.
    """

    if len(papers) < 2:
        return None, "Please upload at least two research papers."

    names = [paper["name"] for paper in papers]

    documents = [
        prepare_text(paper["text"])
        for paper in papers
    ]

    # Check for empty documents
    if any(not document for document in documents):
        return None, "One or more papers contain no readable text."

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_matrix = cosine_similarity(tfidf_matrix)

        return {
            "names": names,
            "similarity": similarity_matrix,
            "features": vectorizer.get_feature_names_out()
        }, None

    except ValueError as error:
        return None, str(error)
