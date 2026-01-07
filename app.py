import os
import re
import random
import sqlite3
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO


from flask import (
    Flask, render_template, request,
    redirect, url_for, jsonify, session, send_file
)

from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
from docx import Document as WordDocument

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Pakai folder lokal untuk data NLTK (penting untuk Railway)
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Cek & download resource yang diperlukan
for resource, locator in [
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(locator)
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR)



app = Flask(__name__)

# ===== CONFIG =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["SECRET_KEY"] = "ganti-secret-key-ini"


# ===== NLP setup =====
STOPWORDS = set(stopwords.words("indonesian"))
STEMMER = StemmerFactory().create_stemmer()

KATA_KERJA_KOGNITIF = {
    "definisi": "Jelaskan",
    "fungsi": "Uraikan",
    "proses": "Jabarkan",
    "why": "Analisis",
}

KONSEP_JAWABAN = [
    "definisi",
    "fungsi",
    "tujuan",
    "manfaat",
    "proses",
    "penyebab",
    "akibat",
    "konsep",
]


# =============================
# UTIL: FILE
# =============================
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_text_from_file(filepath: str) -> str:
    """Extract teks dari PDF/DOCX + hapus daftar isi."""
    try:
        text = ""

        if filepath.lower().endswith(".pdf"):
            reader = PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        elif filepath.lower().endswith(".docx"):
            doc = WordDocument(filepath)
            for p in doc.paragraphs:
                text += p.text + "\n"

        if not text.strip():
            raise ValueError("File tidak mengandung teks")

        # Hapus blok 'DAFTAR ISI' sampai 'BAB I'
        text = re.sub(
            r"DAFTAR ISI(.|\n)*?BAB I",
            "BAB I",
            text,
            flags=re.IGNORECASE,
        )

        return text

    except Exception as e:
        raise Exception(f"Error extract teks: {e}")


# =============================
# NLP
# =============================
def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        STEMMER.stem(w)
        for w in tokens
        if w not in STOPWORDS and len(w) > 3
    ]
    return " ".join(tokens)


def is_kalimat_valid(kalimat: str) -> bool:
    if len(kalimat) < 40:
        return False
    if re.search(r"\d{2,}", kalimat):
        return False
    if re.search(r"^(BAB|DAFTAR|Tabel|Gambar)", kalimat, re.I):
        return False
    if len(kalimat) > 300:
        return False
    return True


def extract_concept(kalimat: str) -> str:
    kalimat_clean = kalimat.lower()

    patterns = [
        r"(?:adalah|merupakan|yaitu)\s+([^,.]+)",
        r"(?:fungsi|tujuan|manfaat)\s+([^,.]+)",
        r"^([^,.\s]+(?:\s+[^,.\s]+)?)\s+(?:adalah|merupakan)",
    ]

    for pattern in patterns:
        m = re.search(pattern, kalimat_clean)
        if m:
            konsep = m.group(1).strip()
            if len(konsep) > 5:
                return konsep

    words = word_tokenize(kalimat)
    for w in words:
        if len(w) > 5 and w.lower() not in STOPWORDS:
            return w.capitalize()

    return "Konsep"


def analyze_text(text: str):
    # Ambil kalimat berurutan dari materi, tidak diacak TF-IDF
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if is_kalimat_valid(s)]

    # Batasi maksimal 30 kalimat pertama supaya tidak terlalu banyak
    return sentences[:30]

    clean = [preprocess_text(s) for s in sentences]
    tfidf = TfidfVectorizer(max_features=500)
    matrix = tfidf.fit_transform(clean)
    scores = matrix.sum(axis=1).A1

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:15]]


# =============================
# GENERATE ESSAY
# =============================
def generate_essay_questions(sentences, jumlah: int):
    hasil = []
    konsep_dipakai = set()

    for kalimat in sentences:
        if len(hasil) >= jumlah:
            break

        # Kalimat sudah difilter di analyze_text, jadi tidak perlu cek ulang terlalu banyak
        konsep = extract_concept(kalimat)
        if not konsep or konsep in konsep_dipakai:
            continue

        konsep_dipakai.add(konsep)
        teks = kalimat.lower()

        # Tentukan tipe soal berdasarkan kata kunci di kalimat YANG SAMA
        if any(w in teks for w in ["adalah", "merupakan", "definisi"]):
            tipe = "definisi"
        elif "fungsi" in teks:
            tipe = "fungsi"
        elif "proses" in teks or "tahap" in teks:
            tipe = "proses"
        else:
            tipe = "why"

        kata_kerja = KATA_KERJA_KOGNITIF[tipe]

        # Jawaban = kalimat yang sama dengan sumber konsep
        jawaban = kalimat
        # (opsional) batasi panjang jawaban biar rapi
        if len(jawaban) > 300:
            jawaban = jawaban[:300] + "..."

        hasil.append(
            {
                "question": f"{kata_kerja} {konsep} berdasarkan materi di atas!",
                "answer": jawaban,
                "type": "essay",
            }
        )

    return hasil

# =============================
# GENERATE PILIHAN GANDA
# =============================
def generate_pg_questions(sentences, jumlah: int):
    hasil = []

    for kalimat in sentences:
        if len(hasil) >= jumlah:
            break

        teks = kalimat.lower()

        if any(
            x in teks
            for x in [
                "silahkan",
                "harap",
                "jawablah",
                "daftar isi",
                "daftar pustaka",
                "tabel",
                "gambar",
                "bab ",
            ]
        ):
            continue

        if len(kalimat.split()) < 8:
            continue

        if "adalah" in teks or "merupakan" in teks:
            jawaban = "definisi"
        elif "fungsi" in teks:
            jawaban = "fungsi"
        elif "tujuan" in teks:
            jawaban = "tujuan"
        elif "akibat" in teks or "dampak" in teks:
            jawaban = "akibat"
        elif "proses" in teks:
            jawaban = "proses"
        else:
            jawaban = "konsep"

        distraktor = list(set(KONSEP_JAWABAN) - {jawaban})
        opsi = random.sample(distraktor, min(3, len(distraktor))) + [jawaban]
        random.shuffle(opsi)

        hasil.append(
            {
                "question": (
                    "Perhatikan pernyataan berikut:\n\n"
                    f"\"{kalimat}\"\n\n"
                    "apa yang dimaksud dari pernyataan tersebut …"
                ),
                "options": opsi,
                "answer": jawaban,
                "type": "pg",
            }
        )

    return hasil


def generate_questions(sentences, jumlah: int, jenis: str):
    if jenis == "essay":
        return generate_essay_questions(sentences, jumlah)
    if jenis == "pg":
        return generate_pg_questions(sentences, jumlah)
    return []


# =============================
# DATABASE (opsional dipakai)
# =============================
def init_db():
    try:
        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                materi TEXT,
                jenis_soal TEXT,
                jumlah_soal INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                pertanyaan TEXT,
                jawaban TEXT,
                jenis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """
        )

        conn.commit()
        conn.close()
    except Exception as e:
        print("DB init error:", e)


# =============================
# ROUTES
# =============================
# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # validasi file
            if "file" not in request.files:
                return render_template("index.html", error="File tidak ditemukan"), 400

            file = request.files["file"]

            if file.filename == "":
                return render_template("index.html", error="File belum dipilih"), 400

            if not allowed_file(file.filename):
                return render_template(
                    "index.html",
                    error="Format file hanya PDF atau DOCX",
                ), 400

            jenis_soal = request.form.get("jenis_soal", "pg")

            try:
                jumlah_soal = int(request.form.get("jumlah_soal", 5))
                if not (1 <= jumlah_soal <= 50):
                    return render_template(
                        "index.html",
                        error="Jumlah soal harus antara 1–50",
                    ), 400
            except ValueError:
                return render_template(
                    "index.html",
                    error="Jumlah soal harus berupa angka",
                ), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            materi = extract_text_from_file(filepath)
            sentences = analyze_text(materi)

            if not sentences:
                os.remove(filepath)
                return render_template(
                    "index.html",
                    error="Tidak ada materi yang bisa diproses",
                ), 400

            questions = generate_questions(sentences, jumlah_soal, jenis_soal)

            if not questions:
                os.remove(filepath)
                return render_template(
                    "index.html",
                    error="Tidak dapat menghasilkan soal dari file tersebut",
                ), 400

            # simpan ke session (untuk halaman result)
            session["materi"] = materi
            session["questions"] = questions
            session["jenis_soal"] = jenis_soal

            # ==============================
            # SIMPAN KE DATABASE
            # ==============================
            conn = sqlite3.connect("database.db")
            c = conn.cursor()

            # simpan 1 baris sesi
            c.execute(
                """
                INSERT INTO sessions (materi, jenis_soal, jumlah_soal)
                VALUES (?, ?, ?)
                """,
                (materi, jenis_soal, jumlah_soal),
            )
            session_id = c.lastrowid

            # simpan semua soal
            for q in questions:
                c.execute(
                    """
                    INSERT INTO questions (session_id, pertanyaan, jawaban, jenis)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, q["question"], q["answer"], q["type"]),
                )

            conn.commit()
            conn.close()
            # ==============================

            os.remove(filepath)

            return redirect(url_for("result"))

        except Exception as e:
            return render_template("index.html", error=f"Error: {e}"), 500

    return render_template("index.html")


@app.route("/result")
def result():
    questions = session.get("questions")
    jenis_soal = session.get("jenis_soal", "pg")
    materi = session.get("materi", "")

    if not questions:
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        questions=questions,
        jenis_soal=jenis_soal,
        total_soal=len(questions),
        materi=materi,
    )


@app.route("/download")
def download():
    questions = session.get("questions")
    jenis_soal = session.get("jenis_soal", "pg")

    if not questions:
        return redirect(url_for("index"))

    # Generate PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Judul
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1f77d2"),
        spaceAfter=20,
        alignment=1,  # center
    )
    title = Paragraph("SOAL UJIAN", title_style)
    story.append(title)

    jenis_text = "Pilihan Ganda (PG)" if jenis_soal == "pg" else "Essay"
    subtitle = Paragraph(f"<b>Jenis:</b> {jenis_text} | <b>Total Soal:</b> {len(questions)}", styles["Normal"])
    story.append(subtitle)
    story.append(Spacer(1, 0.3 * inch))

    # Soal
    for idx, q in enumerate(questions, 1):
        # Nomor dan pertanyaan
        question_text = f"<b>{idx}. {q['question']}</b>"
        story.append(Paragraph(question_text, styles["Normal"]))

        if jenis_soal == "pg":
            # Opsi jawaban
            for i, opt in enumerate(q.get("options", []), 1):
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{chr(96+i)}. {opt}", styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        else:  # essay
            # Tempat jawaban
            story.append(Paragraph("<i>Jawaban:</i>", styles["Normal"]))
            story.append(Spacer(1, 0.5 * inch))

        story.append(Spacer(1, 0.2 * inch))

    # Build PDF
    doc.build(story)
    buffer.seek(0)

    # Return sebagai file download
    return send_file(
        buffer,
        as_attachment=True,
        download_name="soal_ujian.pdf",
        mimetype="application/pdf",
    )



@app.errorhandler(413)
def too_large(e):
    return (
        render_template(
            "index.html",
            error="File terlalu besar (maksimal 50 MB)",
        ),
        413,
    )


@app.errorhandler(500)
def internal_error(e):
    return (
        render_template(
            "index.html",
            error="Terjadi error pada server, coba lagi",
        ),
        500,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
