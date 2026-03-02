# 🤖 Phc Bot - Economy & AI Discord Bot (VN)

Chào mừng cậu đến với **Phc Bot**! Đây là một con bot Discord giải trí tập trung vào hệ thống kinh tế (Economy) thuần Việt, kết hợp với trí tuệ nhân tạo (AI) chạy Local cực chất.

---

## ✨ Các Tính Năng Nổi Bật

### ⛏️ Hệ Thống Khai Thác (Mining)
- **Lệnh:** `p mine`
- **Cơ chế:** Đào quặng với tỉ lệ rơi và chất lượng quặng ngẫu nhiên.
- **Hệ thống Cuốc (Pickaxes):** Nâng cấp cuốc để tăng sản lượng và tỉ lệ gặp quặng hiếm.
  - *Cuốc gỗ, Đá, Sắt, Vàng, Kim cương.*
- **Chất lượng quặng:** Quặng đào được có phần trăm chất lượng (%), ảnh hưởng đến giá trị khi giao dịch.

### 💰 Kinh Tế & Cửa Hàng (Shop & Economy)
- **Lệnh:** `p shop`, `p shop buy [tên_vật_phẩm]`
- **Tiền tệ:** Kesling (Biểu tượng: <:kesling:1434181800539979848>).
- **Mua bán:** Hệ thống alias thông minh giúp cậu mua đồ bằng tiếng Việt không dấu (vd: `cupgo`, `cupkimcuong`).
- **Kho đồ (Inventory):** Lưu trữ quặng, nông sản và trang bị cá nhân.

### 🧠 Trí Tuệ Nhân Tạo (Local AI - Llama-chan)
- **Lệnh:** `p ai [câu hỏi]`
- **Model:** Sử dụng **Llama 3.2 1B Instruct** chạy trực tiếp trên máy chủ.
- **Tính năng đặc biệt:** - **RAG (Retrieval-Augmented Generation):** AI có khả năng tự tra cứu kiến thức nội bộ từ file `knowledge.json`.
  - **Web Search:** Tự động tìm kiếm thông tin mới nhất trên Google/DuckDuckGo nếu cần.
  - **Vibe Wibu:** Trò chuyện cực kỳ lầy lội, xưng hô "Anh/Oniichan" chuẩn phong cách Anime.

---

## 🛠 Cài Đặt & Yêu Cầu Kỹ Thuật

### ⚠️ Nhược Điểm (Lưu Ý Quan Trọng)
Vì bot sử dụng **AI Local**, cậu **bắt buộc phải tải Model về máy** để bot có thể hoạt động. Điều này dẫn đến một số hạn chế:
1. **Dung lượng:** Cần ổ cứng trống để chứa file model `.gguf`.
2. **Tài nguyên:** Ngốn RAM và CPU khi xử lý câu hỏi (Bot được tối ưu cho server tầm 3GB RAM trở lên).
3. **Model yêu cầu:** `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

### Các thư viện cần thiết:
```bash
pip install discord.py llama-cpp-python sentence-transformers faiss-cpu numpy duckduckgo_search python-dotenv
