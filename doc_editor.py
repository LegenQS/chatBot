"""Document editor UI and logic for visualizing/editing the manual."""
import json
import subprocess
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CHUNK_DIR = BASE_DIR / "chunk"
ORIGINAL_DOCX = BASE_DIR / "original" / "error-diagnose-and-maintenance-instruction.docx"
MANUAL_DOCS = BASE_DIR / "manual_docs.json"
IMAGE_DIR = BASE_DIR / "images"


def load_current_docs():
    """Load the parsed document structure from manual_docs.json."""
    if not MANUAL_DOCS.exists():
        return []
    with open(MANUAL_DOCS, "r", encoding="utf-8") as f:
        return json.load(f)


def save_docs_to_json(docs, filepath=MANUAL_DOCS):
    """Save document structure back to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def rebuild_index():
    """Run build_index.py to rebuild the search index."""
    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / "build_index.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def rebuild_chunks():
    """Run parse_doc.py to re-parse the manual and regenerate chunk JSONs."""
    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / "parse_doc.py")],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def list_images():
    """Return list of image files in the images/ directory."""
    if not IMAGE_DIR.exists():
        return []
    return sorted([f.name for f in IMAGE_DIR.glob("*") if f.is_file()])


def organize_docs_by_chapter(docs):
    """Organize flat document list into chapter/section hierarchy."""
    chapters = {}
    for doc in docs:
        path = doc.get("metadata", {}).get("path", "Unknown")
        # Path format: "Chapter X › Section Y › Subsection Z"
        parts = [p.strip() for p in path.split("›")]

        chapter_title = parts[0] if parts else "Unknown"
        section_title = parts[1] if len(parts) > 1 else "Main"

        if chapter_title not in chapters:
            chapters[chapter_title] = {}
        if section_title not in chapters[chapter_title]:
            chapters[chapter_title][section_title] = []

        chapters[chapter_title][section_title].append(doc)

    return chapters


def render_doc_structure(docs):
    """Render interactive document structure with expandable chapters/sections."""
    if not docs:
        st.info("No document content available")
        return

    chapters = organize_docs_by_chapter(docs)

    for chapter_num, (chapter_title, sections) in enumerate(chapters.items(), 1):
        with st.expander(f"📚 {chapter_title}", expanded=False):
            for section_title, docs_in_section in sections.items():
                with st.expander(f"  📄 {section_title}", expanded=False):
                    for doc in docs_in_section:
                        st.markdown(f"**{doc.get('section_title', 'Content')}**")

                        # Instructions
                        if doc.get("instructions"):
                            st.markdown("📝 **Instructions:**")
                            for instr in doc["instructions"]:
                                st.markdown(f"- {instr}")

                        # Warnings
                        if doc.get("warnings"):
                            st.markdown("⚠️ **Warnings:**")
                            for warn in doc["warnings"]:
                                st.markdown(f"- {warn}")

                        # Images
                        if doc.get("metadata", {}).get("images"):
                            st.markdown("🖼️ **Images:**")
                            for img in doc["metadata"]["images"]:
                                img_path = IMAGE_DIR / img
                                if img_path.exists():
                                    st.image(str(img_path), use_container_width=True)

                        st.divider()


def add_new_chapter(docs, chapter_title):
    """Add a new chapter to the document."""
    new_doc = {
        "metadata": {
            "path": chapter_title,
            "images": []
        },
        "section_title": chapter_title,
        "instructions": [],
        "warnings": []
    }
    docs.append(new_doc)
    return docs


def add_new_section(docs, chapter_title, section_title):
    """Add a new section to a chapter."""
    path = f"{chapter_title} › {section_title}"
    new_doc = {
        "metadata": {
            "path": path,
            "images": []
        },
        "section_title": section_title,
        "instructions": [],
        "warnings": []
    }
    docs.append(new_doc)
    return docs


def add_new_paragraph(docs, chapter_title, section_title, paragraph_text, para_type="instruction"):
    """Add a new paragraph (instruction or warning) to a section."""
    path = f"{chapter_title} › {section_title}"

    # Find or create the section
    found = False
    for doc in docs:
        if doc.get("metadata", {}).get("path") == path:
            if para_type == "instruction":
                doc["instructions"].append(paragraph_text)
            else:
                doc["warnings"].append(paragraph_text)
            found = True
            break

    if not found:
        # Create new section if it doesn't exist
        new_doc = {
            "metadata": {
                "path": path,
                "images": []
            },
            "section_title": section_title,
            "instructions": [paragraph_text] if para_type == "instruction" else [],
            "warnings": [paragraph_text] if para_type == "warning" else []
        }
        docs.append(new_doc)

    return docs
