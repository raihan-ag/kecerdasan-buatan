# RAG System for TP-Link Archer C80 User Manual
# Deskripsi
Sistem Retrieval-Augmented Generation (RAG) untuk memberikan jawaban akurat berdasarkan user manual Router WiFi TP-Link Archer C80. Sistem ini mengintegrasikan data ingestion, chunking, embedding, vector database, retrieval, dan generation dengan LLM untuk menghindari halusinasi. Pembaruan terbaru: Integrasi dashboard interaktif menggunakan Streamlit untuk demo penggunaan sistem secara real-time. File rag_router.py telah digabung ke app.py untuk simplifikasi; rag_router.py dapat dihapus jika tidak diperlukan lagi.
Studi kasus ini dibangun untuk Ujian Akhir Semester (UAS) Mata Kuliah Artificial Intelligence (SI148), dengan fokus pada Smart Manual Support (Produk Elektronik). Sistem memenuhi CPMK061, CPMK081, dan CPMK082, serta CPL06 dan CPL08.

# Pembagian Tugas
Raihan: (Data Ingestion, Chunking, dan Integrasi Dashboard)
Raihan: (Embedding dan Vector Database)
Raihan: (Retrieval dan Generation)
Raihan: (Testing, Dokumentasi, dan Video Demo)

# Persyaratan Sistem
Python 3.8+
Koneksi internet untuk download model HuggingFace pertama kali (opsional untuk LLM eksternal seperti OpenAI)

# Instal Dependensi
Jalankan perintah berikut untuk instal library yang diperlukan:
textpip install streamlit transformers sentence-transformers torch numpy scipy

# Cara Run
Pastikan file router_manual.txt (knowledge base dari user manual TP-Link Archer C80) ada di folder yang sama.
Jalankan dashboard interaktif:textstreamlit run app.py
Akses di browser: http://localhost:8501
Masukkan pertanyaan di form, klik "Tanya" untuk lihat jawaban dan context retrieval.

Contoh penggunaan di kode (tanpa dashboard):Pythonfrom app import rag_query  # Import fungsi dari app.py jika perlu
print(rag_query("Bagaimana cara reset konfigurasi pabrik pada router ini?"))

# Dependensi
Streamlit (untuk dashboard UI)
Transformers (HuggingFace) - Untuk embedding dan generation
Sentence-Transformers (model all-MiniLM-L6-v2)
Torch (backend transformer)
NumPy dan SciPy (untuk vector database sederhana)
