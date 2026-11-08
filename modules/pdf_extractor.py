
import pymupdf


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded research paper PDF.
    """

    pdf_document = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    extracted_text = ""

    for page_number, page in enumerate(pdf_document):
        page_text = page.get_text()
        extracted_text += page_text + "\n"

    pdf_document.close()

    return extracted_text
