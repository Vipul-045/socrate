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
    
def chunk_text_with_offsets(text: str) -> list[tuple[str, int]]:
    """Same sliding-window logic as chunk_text, but also returns each chunk's
    character offset in the original text, needed to map chunks -> topics."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        raw = text[start:end]
        stripped = raw.strip()
        if stripped:
            leading_ws = len(raw) - len(raw.lstrip())
            chunks.append((stripped, start + leading_ws))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks