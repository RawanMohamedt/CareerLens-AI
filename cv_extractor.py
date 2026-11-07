from pathlib import Path
import warnings
import docx

from pypdf import PdfReader
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


# =========================================================
# 1. قراءة ملفات PDF (مع استخراج الروابط التشعبية المدمجة /Annots)
# =========================================================

def _extract_pdf_page_links(page) -> list[str]:
    """Extract embedded hyperlink URIs from a PDF page's /Annots dictionary."""
    urls: list[str] = []
    try:
        annots = page.get("/Annots")
        if not annots:
            return urls
        for annot in annots:
            obj = annot.get_object()
            action = obj.get("/A") if obj else None
            if action and "/URI" in action:
                uri = str(action["/URI"]).strip()
                if uri and not uri.lower().startswith("mailto:"):
                    if uri not in urls:
                        urls.append(uri)
    except Exception:
        pass
    return urls


def read_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))

        # التحقق من تشفير الملف
        if reader.is_encrypted:
            raise ValueError(
                "ملف PDF مشفر ولقد تعذر قراءته."
            )

        pages_text = []
        embedded_links: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages_text.append(page_text)
            for link in _extract_pdf_page_links(page):
                if link not in embedded_links:
                    embedded_links.append(link)

        full_text = "\n\n".join(pages_text).strip()

        # إلحاق الروابط المدمجة في حال لم تكن مكتوبة كنص صريح داخل الصفحة
        missing_links = [
            url for url in embedded_links
            if url not in full_text
        ]
        if missing_links:
            full_text = f"{full_text}\n\nEmbedded Links:\n" + "\n".join(missing_links)

        if not full_text.strip():
            warnings.warn(
                f"الملف {file_path.name} فارغ أو يحتوي على صور فقط بدون نص محدد."
            )

        return full_text.strip()

    except Exception as error:
        raise RuntimeError(
            f"حدث خطأ أثناء قراءة ملف PDF: {error}"
        ) from error


# =========================================================
# 2. قراءة ملفات Word (DOCX)
# =========================================================

def _iter_block_items(parent):
    if isinstance(parent, docx.document.Document):
        parent_elm = parent.element.body
    elif hasattr(parent, "_tc"):
        parent_elm = parent._tc
    else:
        raise TypeError("عنصر غير مدعوم لاستخراج الكتل")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def read_docx(file_path: Path) -> str:
    try:
        doc = docx.Document(str(file_path))
        full_element_text = []

        for block in _iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    full_element_text.append(text)

            elif isinstance(block, Table):
                for row in block.rows:
                    row_text = [
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    ]
                    if row_text:
                        full_element_text.append(" | ".join(row_text))

        extracted_text = "\n".join(full_element_text).strip()

        if not extracted_text:
            warnings.warn(
                f"الملف {file_path.name} فارغ أو لا يحتوي على نصوص قابلة للقراءة."
            )

        return extracted_text

    except Exception as error:
        raise RuntimeError(
            f"حدث خطأ أثناء قراءة ملف DOCX: {error}"
        ) from error


# =========================================================
# 3. قراءة ملفات النص TXT
# =========================================================

def read_txt(file_path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp1256", "latin-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read().strip()
                if not content:
                    warnings.warn(f"الملف {file_path.name} فارغ.")
                return content
        except UnicodeDecodeError:
            continue
        except Exception as error:
            raise RuntimeError(
                f"حدث خطأ أثناء قراءة ملف TXT: {error}"
            ) from error

    raise ValueError(
        f"تعذر تحديد ترميز الملف {file_path.name} باستخدام الترميزات المدعومة."
    )


# =========================================================
# 4. الدالة الرئيسية لاستخراج النص حسب امتداد الملف
# =========================================================

def extract_text_from_cv(file_path: str | Path) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"الملف غير موجود في المسار: {path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return read_pdf(path)
    elif extension == ".docx":
        return read_docx(path)
    elif extension == ".txt":
        return read_txt(path)
    else:
        raise ValueError(
            f"صيغة الملف '{extension}' غير مدعومة. الصيغ المدعومة هي: .pdf, .docx, .txt"
        )
