from pathlib import Path
import warnings
import docx

from pypdf import PdfReader
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


# =========================================================
# 1. استخراج النص من PDF
# =========================================================

def read_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))

        # التحقق من تشفير الملف
        if reader.is_encrypted:
            raise ValueError(
                "ملف PDF مشفر. يجب توفير كلمة المرور أولاً."
            )

        pages_text = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages_text.append(page_text)
            else:
                # قد تكون الصفحة صورة أو صفحة فارغة
                warnings.warn(
                    f"لم يتم استخراج نص من الصفحة {page_number} "
                    f"في الملف: {file_path.name}"
                )

        result = "\n".join(pages_text)

        if not result.strip():
            warnings.warn(
                "لم يتم استخراج أي نص. قد يحتاج الملف إلى OCR."
            )

        return result

    except (ValueError, RuntimeError):
        raise

    except Exception as e:
        raise RuntimeError(
            f"خطأ أثناء قراءة PDF '{file_path.name}': {e}"
        ) from e


# =========================================================
# 2. المرور على فقرات وجداول Word بترتيب ظهورها
# =========================================================

def iter_docx_blocks(parent):
    """
    المرور على الفقرات والجداول حسب ترتيبها في المستند.
    يدعم المستند وخلايا الجداول.
    """
    if hasattr(parent, "element") and hasattr(parent.element, "body"):
        parent_element = parent.element.body
    elif hasattr(parent, "_tc"):
        parent_element = parent._tc
    else:
        parent_element = parent.element

    for child in parent_element.iterchildren():

        if isinstance(child, CT_P):
            yield Paragraph(child, parent)

        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_docx_content(parent, output):
    """
    استخراج الفقرات والجداول، مع محاولة دعم
    الجداول المتداخلة داخل خلايا الجداول.
    """
    for block in iter_docx_blocks(parent):

        if isinstance(block, Paragraph):
            paragraph_text = block.text.strip()

            if paragraph_text:
                output.append(paragraph_text)

        elif isinstance(block, Table):
            for row in block.rows:
                row_cells = []

                for cell in row.cells:
                    cell_parts = []

                    extract_docx_content(
                        cell,
                        cell_parts
                    )

                    cell_text = " ".join(
                        part.strip()
                        for part in cell_parts
                        if part.strip()
                    )

                    row_cells.append(cell_text)

                # الحفاظ على حدود الصفوف والخلايا
                if any(row_cells):
                    output.append(" | ".join(row_cells))


def read_docx(file_path: Path) -> str:
    try:
        document = docx.Document(str(file_path))
        text_parts = []

        extract_docx_content(
            document,
            text_parts
        )

        return "\n".join(text_parts)

    except Exception as e:
        raise RuntimeError(
            f"خطأ أثناء قراءة Word '{file_path.name}': {e}"
        ) from e


# =========================================================
# 3. استخراج النص من TXT مع التعامل مع الترميز
# =========================================================

def read_txt(file_path: Path) -> str:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1256",
        "cp1252",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return file_path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(
        f"تعذر تحديد ترميز ملف TXT '{file_path.name}'. "
        f"آخر خطأ: {last_error}"
    )


# =========================================================
# 4. خريطة امتدادات الملفات
# =========================================================

HANDLERS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".txt": read_txt,
}


# =========================================================
# 5. الدالة الرئيسية لاستخراج النص
# =========================================================

def extract_text(file_path_str: str) -> str:
    file_path = Path(file_path_str)

    # التحقق من وجود الملف
    if not file_path.exists():
        raise FileNotFoundError(
            f"الملف غير موجود: {file_path}"
        )

    # التأكد أن المسار يشير إلى ملف
    if not file_path.is_file():
        raise ValueError(
            f"المسار لا يشير إلى ملف: {file_path}"
        )

    # تحديد نوع الملف
    ext = file_path.suffix.lower()
    handler = HANDLERS.get(ext)

    if handler is None:
        supported = ", ".join(HANDLERS.keys())

        raise ValueError(
            f"امتداد غير مدعوم: {ext or 'بدون امتداد'}. "
            f"الامتدادات المدعومة: {supported}"
        )

    # استخراج النص
    text = handler(file_path)

    return text


# =========================================================
# 6. تجربة الكود
# =========================================================

if __name__ == "__main__":

    cv_path = "my_cv.docx"

    try:
        raw_text = extract_text(cv_path)

        if raw_text.strip():
            print("تم استخراج النص بنجاح.")
            print("-" * 50)
            print(raw_text)
        else:
            print(
                "الملف موجود، لكن لم يتم استخراج نص منه."
            )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"حدث خطأ: {e}")