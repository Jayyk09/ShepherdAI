from PyPDF2 import PdfReader


def extract_text_from_pdfs(pdf_paths):
    """
    Extract text from multiple PDF files.
    Args:
        pdf_paths (list of str): List of PDF file paths.
    Returns:
        list: List of raw text strings from all PDFs.
    """
    pdf_texts = []

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        texts = [p.extract_text().strip() for p in reader.pages]
        pdf_texts.extend(text for text in texts if text)  # Filter empty strings

    return pdf_texts