# Game Toxic Detection 🎮🚫

**Joint Intent Classification & Slot Filling for Toxic Game Chat Understanding**

Proyek ini merupakan implementasi model **Slot-Gated Joint Natural Language Understanding** (Goo et al., 2018) untuk menangani dua *task* utama pada percakapan game online yang bersifat toksik:

1.  **Intent Classification**: Menentukan kategori toksisitas utama dalam satu *utterance* (kalimat).
2.  **Slot Filling**: Menandai token spesifik yang mengandung unsur toksik (penghinaan, sarkasme, *flame*, dll).

Dataset yang digunakan adalah bagian dari **CONDA Shared Task**, yang berisi log percakapan dari *multiplayer online matches*.

---

## 🌐 1. Dataset Overview

Dataset terdiri dari chat pemain dengan struktur data sebagai berikut:
* **Tokenized text**: Kalimat input yang telah dipecah.
* **Intent Label**: Kategori toksisitas global (`E`, `I`, `A`, `O`).
* **Slot Labels**: Label per-token (`T`, `C`, `D`, `S`, `P`, `O`).
* **Metadata**: `matchId`, `conversationId`, `timestamps`.

**Karakteristik Data:**
* Distribusi Intent **O (Other/Non-toxic)** sangat mendominasi (*high class imbalance*).
* Slot label **S (Slang)** dan **T (Toxic)** memiliki peran krusial dalam mendeteksi pola toksisitas.

---

## 🧠 2. Model Architecture: Slot-Gated Joint NLU

Model ini mengadopsi arsitektur **Slot-Gated Intent Attention** sesuai paper Goo et al. (2018). Model tidak menggunakan *full slot attention*, melainkan fokus pada *intent attention* untuk memandu prediksi slot.

### Komponen Utama:

1.  **BiLSTM Encoder**
    Menghasilkan representasi kontekstual level token ($h_1 \dots h_n$).

2.  **Intent Attention**
    Membangun context vector global ($c^I$) melalui mekanisme *additive attention*.

3.  **Slot Gate (Intent Gating)** 🔑
    Menggabungkan informasi lokal ($h_i$) dan global ($c^I$) untuk memperkuat prediksi slot. Gate ini belajar seberapa besar pengaruh *intent* global terhadap *slot* tertentu.

    $$g_i = v^T \tanh(h_i + W \cdot c^I)$$

4.  **Prediction**
    $$y_i^S = \text{softmax}(W(h_i + g_i \cdot c^I))$$

---

## ⚙️ 3. Training Setup

Perbandingan konfigurasi antara model **Baseline** dan model **Optimized (Augmented)**.

| Parameter | Baseline | Optimized |
| :--- | :--- | :--- |
| **Epochs** | 10 | 10 |
| **Batch Size** | 32 | 16 |
| **Learning Rate** | `1e-3` | `1e-4` |
| **Hidden Dim** | 128 | 128 |
| **Embedding Dim** | 200 | 200 (Word2Vec) |
| **Dropout** | 0.4 | 0.4 |
| **Regularization** | None | Weight Decay `1e-3` |
| **Extras** | - | Class Weighting, Synonym Augmentation |

---

## 🔀 4. Workflow Summary

1.  **Preprocessing**: Token alignment, Label mapping, Padding & Masking.
2.  **Forward Pass**: Encoding → Attention → Gating → Predictions.
3.  **Joint Loss Computation**:
    $$\mathcal{L}_{total} = \mathcal{L}_{intent} + \mathcal{L}_{slot}$$
4.  **Evaluation**: Menggunakan metrik **JSA (Joint Semantic Accuracy)**, F1-Intent, dan F1-Slot.
5.  **Inference**: Generate `answer_test.txt` untuk submission.

---

## 📊 5. Experimental Results

Hasil menunjukkan bahwa **Baseline** memiliki performa keseluruhan (JSA) yang lebih stabil dibandingkan versi Optimized, meskipun versi Optimized unggul dalam mendeteksi slot tertentu.

### Performance Metrics

| Metric | Baseline (Best) 🏆 | Optimized (Best) |
| :--- | :--- | :--- |
| **JSA (Joint Semantic Accuracy)** | **0.8726** | 0.8685 |
| **F1 Intent - E (Explicit)** | **0.831** | 0.820 |
| **F1 Intent - I (Implicit)** | **0.719** | 0.660 |
| **F1 Slot - T (Toxic)** | 0.966 | 0.960 |
| **F1 Slot - D (Defense)** | 0.933 | **0.950** |
| **F1 Slot - S (Slang)** | 0.988 | **0.990** |

> *Catatan: Data validasi final Baseline menunjukkan JSA ~0.864, sedangkan Augmented ~0.850.*

---

## 🧩 6. Analysis: Why Baseline Outperforms Optimized?

Meskipun model *Optimized* menggunakan teknik *advanced* (Pretrained Word2Vec, Augmentasi Sinonim, Regularisasi), **Baseline tetap unggul dalam metrik JSA**. Berikut analisis penyebabnya:

1.  **Sensitivitas JSA terhadap Intent**:
    JSA mengharuskan prediksi Intent DAN seluruh Slot benar sekaligus. Optimized berhasil meningkatkan F1-Slot, namun penurunan performa pada F1-Intent (terutama kelas *Implicit Attack*) menyebabkan JSA total turun.

2.  **Noise dari Augmentasi Data**:
    Teknik *Synonym Replacement* sering mengubah makna pragmatik dalam konteks *gaming slang*. Kata kasar yang diganti sinonim umum bisa kehilangan konteks toksiknya, membingungkan model.

3.  **Domain Mismatch pada Word2Vec**:
    Embedding Word2Vec standar dilatih pada teks umum (Wikipedia/News), sehingga representasi vektornya kurang menangkap nuansa bahasa *toxic gaming* yang penuh singkatan dan slang unik.

4.  **Hyperparameter yang Terlalu Konservatif**:
    Kombinasi *Learning Rate* kecil (`1e-4`) dan *Weight Decay* membuat model Optimized kesulitan mempelajari representasi intent yang kompleks secara efektif dibandingkan Baseline yang lebih agresif.

---

## 📂 File Structure

* `SlotGatedRNN_Baseline.ipynb`: Kode training dan evaluasi untuk model Baseline.
* `SlotGatedRNN_Augmented+Weighted.ipynb`: Kode eksperimen dengan augmentasi, class weighting, dan Word2Vec.
* `README.md`: Dokumentasi proyek.

---

### References
* Goo, C. W., et al. (2018). *Slot-Gated Modeling for Joint Slot Filling and Intent Prediction*. NAACL.
* CONDA Shared Task Dataset.
