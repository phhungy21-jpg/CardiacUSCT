"""Sets the pandoc default reference.docx's styles to Times New Roman 11pt
body text (with proportionally scaled headings), for preprint manuscript
conversion."""

from docx import Document
from docx.shared import Pt

TEMPLATE_PATH = "preprint_package/_default_reference.docx"

HEADING_SIZES = {
    "Title": 18,
    "Heading 1": 14,
    "Heading 2": 12.5,
    "Heading 3": 11.5,
}


def set_font(style, name: str, size_pt: float) -> None:
    style.font.name = name
    style.font.size = Pt(size_pt)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")
    if rfonts is None:
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    from docx.oxml.ns import qn
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)


def main() -> None:
    doc = Document(TEMPLATE_PATH)

    set_font(doc.styles["Normal"], "Times New Roman", 11)
    for style_name, size in HEADING_SIZES.items():
        if style_name in doc.styles:
            set_font(doc.styles[style_name], "Times New Roman", size)

    for style_name in ("Body Text", "First Paragraph", "Compact", "Block Text"):
        if style_name in doc.styles:
            set_font(doc.styles[style_name], "Times New Roman", 11)

    doc.save(TEMPLATE_PATH)
    print(f"Updated {TEMPLATE_PATH}: Normal + headings set to Times New Roman")


if __name__ == "__main__":
    main()
