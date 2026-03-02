# 🤖 Phc Bot - Vietnam Economy & Local AI Discord Bot

Chào mừng cậu đến với **Phc Bot**! Đây là một con bot Discord đa năng, kết hợp giữa hệ thống kinh tế (Economy) cày cuốc cực cuốn và trí tuệ nhân tạo (AI) chạy Local riêng tư, bảo mật.

---

## ✨ Các Tính Năng Chính

### ⛏️ 1. Hệ Thống Kinh Tế & Mini-game (Mining)
Đây là tính năng cốt lõi giúp các thành viên trong server tương tác và làm giàu:
- **Khai thác (Mine):** Sử dụng lệnh `p mine` để đào quặng. Mỗi loại quặng có tỷ lệ xuất hiện và giá trị khác nhau.
- **Hệ thống Cuốc (Pickaxes):** Nâng cấp từ Cuốc Gỗ lên Cuốc Kim Cương để tăng hiệu suất và đào được vật phẩm hiếm.
- **Chất lượng vật phẩm:** Quặng đào được có chỉ số chất lượng (%), ảnh hưởng trực tiếp đến giá bán.
- **Cửa hàng (Shop):** Mua sắm trang bị và vật phẩm hỗ trợ bằng đơn vị tiền tệ **Kesling**.
- **Game show:** Tích hợp trò chơi "Ai là triệu phú" với mức thưởng hấp dẫn.

### 🧠 2. Trí Tuệ Nhân Tạo Local (Llama-chan)
Không cần API đắt đỏ, bot sở hữu bộ não AI chạy ngay trên máy chủ của cậu:
- **Model:** Sử dụng `Llama 3.2 1B Instruct` (hoặc các dòng GGUF tương đương).
- **RAG (Retrieval-Augmented Generation):** AI có khả năng đọc hiểu dữ liệu từ file `knowledge.json` để trả lời các kiến thức chuyên biệt.
- **Tra cứu Web:** Nếu không biết, bot có thể tự "lướt web" qua DuckDuckGo để cập nhật thông tin mới nhất.
- **Tính cách (Persona):** Được thiết lập với phong cách "Llama-chan" lầy lội, wibu, hay gọi người dùng là "Oniichan".

---

## ⚠️ Nhược Điểm & Lưu Ý Quan Trọng

Cậu cần lưu ý một số điểm sau để bot hoạt động trơn tru:

1. **Phải tải Model thủ công:** Đây là nhược điểm lớn nhất. Bot không đi kèm sẵn "não". Cậu cần tải file model có đuôi `.gguf` (ví dụ: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`) và bỏ vào thư mục bot.
2. **Tốn tài nguyên phần cứng:** Vì chạy AI Local nên bot sẽ ngốn RAM và CPU khi xử lý câu hỏi. Khuyến nghị chạy trên VPS có tối thiểu 2-4GB RAM.
3. **Cài đặt thư viện:** Cần cài đặt `llama-cpp-python` và các thư viện bổ trợ về xử lý ngôn ngữ (SentenceTransformers, FAISS).
4. **là hallucion** chưa fix được.
---

## 🛠 Hướng Dẫn Cài Đặt Nhanh

1. **Cài đặt thư viện:**
   ```bash
   pip install discord.py llama-cpp-python sentence-transformers faiss-cpu python-dotenv
### 1. Cấu hình môi trường
Tạo file `.env` trong thư mục gốc và dán nội dung sau:
```env
TOKEN=your_discord_bot_token_here
LLAMA_MODEL_PATH=đường/dẫn/đến/file/model.gguf
```
### khởi chạy bot
```bash
python app1.py
```
#chúc cậu vọc cái này vui vẻ!
