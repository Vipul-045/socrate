# Splits the big text blob into smaller overlapping pieces
# Overlap ensures context isn't lost at chunk boundaries

from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:                  # skip empty chunks
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP  # slide with overlap

    return chunks