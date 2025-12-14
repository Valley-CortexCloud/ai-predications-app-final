from pathlib import Path

# Centralized cache directory for historical price caches
CACHE_DIR = Path('data_cache')
CACHE_DIR.mkdir(exist_ok=True)
