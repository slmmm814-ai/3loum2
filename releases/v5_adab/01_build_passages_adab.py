# -*- coding: utf-8 -*-
"""
Phase 1: تفكيك النص من مستوى الصفحة إلى مستوى الفقرة/الجملة.
ينشئ جدول `passages` + فهرس FTS5 مرافق `passages_fts`، مع الحفاظ على الربط
الكامل بالكتاب/الصفحة/المجلد/رقم الصفحة المطبوعة لكل فقرة.
"""
import sqlite3, re, sys, html

DB = "فهرس_الادب_part1_sqlite3"

SPAN_RE = re.compile(r"<span[^>]*>.*?</span>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\؟])\s+")

def strip_tags_keep_titles(text):
    """يحذف الوسوم الشكلية لكن يحتفظ بنص العناوين كفقرة عنوان منفصلة."""
    titles = []
    def _cap(m):
        inner = TAG_RE.sub("", m.group(0))
        titles.append(inner.strip())
        return "\u0001"  # مؤشر مكان العنوان
    cleaned = SPAN_RE.sub(_cap, text)
    cleaned = TAG_RE.sub("", cleaned)
    return html.unescape(cleaned), titles

def split_into_passages(raw_text, min_chars=40, max_chars=600):
    """
    يقسم نص الصفحة إلى فقرات (بحسب الأسطر)، ثم يقسّم الفقرات الطويلة إلى
    جمل، ويدمج الفقرات القصيرة جدًا مع ما يليها لتفادي شذرات بلا فائدة.
    يعيد قائمة نصوص (فقرة واحدة لكل عنصر) بترتيبها داخل الصفحة.
    """
    cleaned, titles = strip_tags_keep_titles(raw_text)
    lines = [l.strip() for l in cleaned.split("\n")]
    lines = [l for l in lines if l and l != "\u0001"]

    passages = []
    for t in titles:
        if t:
            passages.append(("title", t))

    buf = ""
    for line in lines:
        if len(line) > max_chars:
            # فقرة طويلة: قسّمها لجمل
            if buf:
                passages.append(("para", buf))
                buf = ""
            sentences = SENT_SPLIT_RE.split(line)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) < max_chars:
                    chunk = (chunk + " " + s).strip()
                else:
                    if chunk:
                        passages.append(("sent", chunk))
                    chunk = s
            if chunk:
                passages.append(("sent", chunk))
        else:
            if len(buf) + len(line) < max_chars:
                buf = (buf + " " + line).strip()
            else:
                if buf:
                    passages.append(("para", buf))
                buf = line
    if buf:
        passages.append(("para", buf))

    # دمج الشذرات الأقصر من min_chars مع الفقرة التالية
    merged = []
    carry = ""
    carry_type = "para"
    for typ, txt in passages:
        if carry:
            txt = (carry + " " + txt).strip()
            typ = carry_type if carry_type != "title" else typ
            carry = ""
        if len(txt) < min_chars and typ != "title":
            carry, carry_type = txt, typ
            continue
        merged.append((typ, txt))
    if carry:
        merged.append((carry_type, carry))
    return merged


def main():
    db = sqlite3.connect(DB)
    cur = db.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS passages;
    CREATE TABLE passages (
        passage_id INTEGER PRIMARY KEY,
        book_id INTEGER,
        book_name TEXT,
        pg INTEGER,
        vol TEXT,
        printed_page TEXT,
        para_idx INTEGER,
        passage_type TEXT,
        text TEXT
    );
    """)

    cur.execute("SELECT rowid, text_orig, book_id, book_name, pg, vol, printed_page FROM pages_fts")
    rows = cur.fetchall()
    print(f"عدد الصفحات المصدر: {len(rows)}", file=sys.stderr)

    insert_buf = []
    pid = 1
    for rowid, text_orig, book_id, book_name, pg, vol, printed_page in rows:
        if not text_orig:
            continue
        passages = split_into_passages(text_orig)
        for idx, (typ, txt) in enumerate(passages):
            insert_buf.append((pid, book_id, book_name, pg, vol, printed_page, idx, typ, txt))
            pid += 1
        if len(insert_buf) >= 5000:
            cur.executemany(
                "INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?)", insert_buf
            )
            insert_buf = []
    if insert_buf:
        cur.executemany("INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?)", insert_buf)

    db.commit()
    cur.execute("SELECT COUNT(*) FROM passages")
    total = cur.fetchone()[0]
    print(f"إجمالي الفقرات/الجمل المستخرجة: {total}", file=sys.stderr)

    # فهرس FTS5 (trigram) على مستوى الفقرة لدعم BM25 دقيق
    cur.executescript("""
    DROP TABLE IF EXISTS passages_fts;
    CREATE VIRTUAL TABLE passages_fts USING fts5(
        text, book_name UNINDEXED, passage_id UNINDEXED, tokenize='trigram'
    );
    """)
    cur.execute("SELECT passage_id, text, book_name FROM passages")
    data = [(t, bn, pid) for pid, t, bn in cur.fetchall()]
    cur.executemany("INSERT INTO passages_fts(text, book_name, passage_id) VALUES (?,?,?)", data)
    db.commit()

    cur.execute("CREATE INDEX IF NOT EXISTS idx_passages_book ON passages(book_id, pg)")
    db.commit()
    print("تم بناء passages + passages_fts بنجاح.", file=sys.stderr)


if __name__ == "__main__":
    main()
