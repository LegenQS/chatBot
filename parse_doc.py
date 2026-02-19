import re
import json
import os
from docx import Document
from docx.oxml.ns import qn

# ---------- CONFIG ----------
INPUT_DOCX = "original/heating-system-checking-instruction.docx"
OUTPUT_JSON = "chunk/heating-system-checking-instruction-chunk.json"
IMAGE_DIR = "images"

DOCUMENT_ID = "tempering_machine_manual_v1"
DOCUMENT_TITLE = "Tempering Machine Operation Manual"
LANGUAGE = "zh-CN"

CHAPTER_RE = re.compile(r'^([一二三四五六七八九十]+)[、﹐，]?(.+)$')

# Section level
SECTION_RE = re.compile(r'^(\d+)、\s*(.+)')

WARNING_KEYWORDS = ["注意", "警告", "禁止", "危险", "⚠"]

# ---------------------------


def ensure_dirs():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


def extract_images(doc):
    """
    Extract images from docx and return mapping of rId -> filename
    """
    image_map = {}

    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_data = rel.target_part.blob
            image_name = rel.target_ref.split("/")[-1]
            image_path = os.path.join(IMAGE_DIR, image_name)

            with open(image_path, "wb") as f:
                f.write(image_data)

            image_map[rel.rId] = image_name

    return image_map


def paragraph_images(paragraph, image_map):
    """
    Return list of image filenames attached to this paragraph.
    """
    images = []

    # Find all <a:blip> elements (these hold image rIds)
    blips = paragraph._element.xpath('.//a:blip')

    for blip in blips:
        rId = blip.get(qn('r:embed'))
        if rId in image_map:
            img_name = image_map[rId]
            if img_name not in images:
                images.append(img_name)

    return images


def main():
    ensure_dirs()

    doc = Document(INPUT_DOCX)
    image_map = extract_images(doc)

    result = {
        "document_id": DOCUMENT_ID,
        "document_title": DOCUMENT_TITLE,
        "language": LANGUAGE,
        "chapters": []
    }

    current_chapter = None
    current_chunk = None
    chunk_counter = 0

    section_stack = []
    for p in doc.paragraphs:
        text = p.text.strip()
        img_in_para = paragraph_images(p, image_map)
        # Skip paragraph only if truly empty
        if not text and not img_in_para:
            continue

        # ---- IMAGE (attach to current chunk if exists) ----
        if img_in_para and current_chunk:
            current_chunk["visual_reference"]["required"] = True
            for img in img_in_para:
                if img not in current_chunk["visual_reference"]["image_ids"]:
                    print(img)
                    current_chunk["visual_reference"]["image_ids"].append(img)

        # ---- CHAPTER ----
        chap_match = CHAPTER_RE.match(text)
        if chap_match:
            chunk_counter = 0
            current_chapter = {
                "chapter_id": f"ch{len(result['chapters']) + 1:02}",
                "chapter_number": chap_match.group(1),
                "chapter_title": chap_match.group(2).strip(),
                "chunks": []
            }
            result["chapters"].append(current_chapter)
            current_chunk = None
            continue

        # ---- SECTION / CHUNK ----
        sec_match = SECTION_RE.match(text)
        if sec_match and current_chapter:

            # Detect indentation level
            raw_text = p.text
            indent = len(raw_text) - len(raw_text.lstrip())

            # Simple rule: 0–2 spaces = level 1, 3+ = level 2
            if indent <= 2:
                level = 1
            else:
                level = 2

            new_section = {
                "section_number": sec_match.group(1),
                "section_title": sec_match.group(2).strip(),
                "instructions": [],
                "warnings": [],
                "visual_reference": {
                    "required": False,
                    "purpose": "",
                    "description": "",
                    "image_ids": []
                },
                "subsections": []
            }

            # Maintain hierarchy stack
            while len(section_stack) >= level:
                section_stack.pop()

            if section_stack:
                section_stack[-1]["subsections"].append(new_section)
            else:
                current_chapter.setdefault("sections", []).append(new_section)

            section_stack.append(new_section)
            current_chunk = new_section
            continue

        # Ignore text before first chapter/section
        if not current_chunk:
            continue

        # ---- WARNING ----
        if any(k in text for k in WARNING_KEYWORDS):
            current_chunk["warnings"].append(text)
            continue

        # ---- INSTRUCTION ----
        current_chunk["instructions"].append(text)

    # ---------- VALIDATION ----------
    for ch in result["chapters"]:
        for c in ch["chunks"]:
            if not c["instructions"]:
                merged_text = c.get("section_title", "")
                if merged_text:
                    ch.setdefault("chapter_instructions", [])
                    ch["chapter_instructions"].append(merged_text)

                # Remove the empty chunk from the list
                ch["chunks"].remove(c)

            if c["visual_reference"]["required"] and not c["visual_reference"]["image_ids"]:
                print(f"⚠ Warning: visual required but no images in {c['chunk_id']}")

    # ---------- SAVE ----------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✔ Parsed {len(result['chapters'])} chapters")
    print(f"✔ JSON saved to {OUTPUT_JSON}")
    print(f"✔ Images saved to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
