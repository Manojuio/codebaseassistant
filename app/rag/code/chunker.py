from tree_sitter_language_pack import get_parser
from pathlib import Path
from dataclasses import dataclass


# -------------------------
# Data Model
# -------------------------

@dataclass
class CodeChunk:
    id: int
    language: str
    file: str
    type: str
    name: str
    text: str
    start_line: int
    end_line: int


# -------------------------
# Configuration
# -------------------------

SUPPORTED_EXTENSIONS = {
    ".js": "javascript",
    ".ts": "typescript",
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".md": "markdown",
    ".json": "json",
}

CHUNK_NODES = {
    "javascript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    ],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    ],
    "python": [
        "function_definition",
        "class_definition",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "constructor_declaration",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
    ],
}

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".venv",
    "venv",
    "tests",
    "test",
    "__tests__",
}

MAX_FILE_SIZE = 300_000  # 300 KB


# -------------------------
# Parser Cache
# -------------------------

PARSER_CACHE = {}

def get_cached_parser(language: str):
    if language not in PARSER_CACHE:
        PARSER_CACHE[language] = get_parser(language)
    return PARSER_CACHE[language]


# -------------------------
# Markdown + JSON Handling
# -------------------------

def chunk_full_file(source: str, language: str, relative_path: str):
    lines = source.splitlines()
    return [
        CodeChunk(
            id=0,
            language=language,
            file=relative_path,
            type="file",
            name="full_file",
            text=source,
            start_line=1,
            end_line=len(lines),
        )
    ]


# -------------------------
# AST Chunking
# -------------------------

def chunk_code_ast(source: str, language: str, relative_path: str):
    parser = get_cached_parser(language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    target_types = set(CHUNK_NODES.get(language, []))
    chunks = []
    chunk_id = 0

    def walk(node):
        nonlocal chunk_id

        if node.type in target_types:
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")

            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    language=language,
                    file=relative_path,
                    type=node.type,
                    name="auto",
                    text=text,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
            chunk_id += 1
            return

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


# -------------------------
# File-Level Processing
# -------------------------

def chunk_file(path: Path, repo_root: Path):
    language = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if not language:
        return []

    if path.stat().st_size > MAX_FILE_SIZE:
        return []

    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return []

    relative_path = str(path.relative_to(repo_root))

    # Markdown → full file
    if language == "markdown":
        return chunk_full_file(source, language, relative_path)

    # Only index root package.json
    if language == "json":
        if path.name == "package.json":
            return chunk_full_file(source, language, relative_path)
        return []

    # Code → AST
    return chunk_code_ast(source, language, relative_path)


# -------------------------
# Directory Traversal
# -------------------------

def chunk_directory(directory: str):
    repo_root = Path(directory)
    all_chunks = []

    for path in repo_root.rglob("*.*"):

        if not path.is_file():
            continue

        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue

        if any(part.startswith(".") for part in path.parts):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        file_chunks = chunk_file(path, repo_root)
        all_chunks.extend(file_chunks)

    return all_chunks