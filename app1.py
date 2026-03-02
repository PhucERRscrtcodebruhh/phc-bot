import discord
from typing import Optional, List
import random
import json
import os
import time
import datetime
from discord.ext import commands,tasks
from discord.ui import View, Button, button
from discord import ButtonStyle 
import math # Cần thiết cho việc tính tổng số trang
import aiohttp
import html
import asyncio
import g4f
import re
from llama_cpp import Llama
# --- RAG Nội Sinh (FAISS + Embedding) - Oniichan dùng faiss-cpu cho server 3GB RAM ---
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

#------------------>.<------------

import dotenv
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('TOKEN')










# --- RAG Trên Mạng (DuckDuckGo) ---
try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False
# --- AI Model: Llama 3.2 1B Instruct (Oniichan dùng Llama thay Qwen) ---
AI_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/home/container/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
if not os.path.exists(AI_MODEL_PATH):
    AI_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
AI_NAME = "Llama-chan"  # Tên AI mới — thay tên đổi họ

print("🌸 Đang nạp bộ não AI Local (Llama-chan) vào app1...")
llm = Llama(
    model_path=AI_MODEL_PATH,
    n_ctx=2048,      # Để context thấp cho nhẹ RAM
    n_threads=2,    # Tối ưu 200% CPU của cậu
    verbose=False
)
print("✅ AI (Llama-chan) đã sẵn sàng trong app1!")


_embedding_model = None   
_faiss_index = None       
_chunk_texts: List[str] = []  
_vector_db_ready = False

owner_id = "1135806949527670835"
subowner_id =["1138020979348606996"]
KESLING_ICON = "<:kesling:1434181800539979848>"
# 🐧 may cai nay la ming seng da vibe coding nhe :))
DATA_FILE = 'data.json'
DEFAULT_BET = 100
PRIZE_TIERS = [100, 200, 300, 500, 1000, 2000, 4000, 8000, 16000, 32000]  # example 10-level prizes

def load_data():
    """Tải dữ liệu người chơi từ file data.json."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Lỗi đọc file data.json. Khởi tạo dữ liệu rỗng.")
                return {}
    return {}

def save_data(data):
    """Lưu dữ liệu người chơi vào file data.json."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

player_inventory = load_data()

def get_user_data(user_id):
    """Lấy dữ liệu của người dùng, nếu chưa có thì tạo mới."""
    user_id_str = str(user_id)
    if user_id_str not in player_inventory:
        player_inventory[user_id_str] = {
            'inventory': {},
            'money': 0
        }
        
        save_data(player_inventory) 
    return player_inventory[user_id_str]



def update_player_money(user_id: str, amount: int):
    player = get_user_data(user_id)
    player["money"] = player.get("money", 0) + amount
    save_data(player_inventory)
    return player["money"]


# Inventory helpers to support quality buckets per ore
def get_total_ore_count(inv, ore_name):
    """Return total count for ore_name considering quality buckets or legacy int."""
    val = inv.get(ore_name)
    if val is None:
        return 0
    if isinstance(val, dict):
        return sum(v for v in val.values())
    try:
        return int(val)
    except Exception:
        return 0


def add_ore_with_quality(inv, ore_name, quality_percent: int, qty: int = 1):
    """Add qty ores at given quality percent (store qualities as strings)."""
    if ore_name not in inv or not isinstance(inv[ore_name], dict):
        # convert legacy int to dict if needed
        old = inv.get(ore_name)
        if old is None:
            inv[ore_name] = {}
        else:
            inv[ore_name] = {"100": int(old)}

    qk = str(int(quality_percent))
    inv[ore_name][qk] = inv[ore_name].get(qk, 0) + qty


def remove_ore_units(inv, ore_name, amount, strategy='highest'):
    """Remove `amount` units of ore_name from inv using strategy 'highest' or 'lowest'.
    Returns dict of removed {quality:count}.
    """
    removed = {}
    val = inv.get(ore_name)
    if val is None:
        return removed

    # if legacy int, treat as 100% quality
    if not isinstance(val, dict):
        available = int(val)
        take = min(available, amount)
        remaining = available - take
        if remaining > 0:
            inv[ore_name] = remaining
        else:
            del inv[ore_name]
        removed['100'] = take
        return removed

    # dict case
    qualities = sorted([int(k) for k in val.keys()])
    if strategy == 'highest':
        qualities = sorted(qualities, reverse=True)
    # iterate
    need = amount
    for q in qualities:
        if need <= 0:
            break
        k = str(q)
        have = val.get(k, 0)
        if have <= 0:
            continue
        take = min(have, need)
        removed[k] = removed.get(k, 0) + take
        val[k] = have - take
        need -= take
        if val[k] == 0:
            del val[k]
    # clean up if empty
    if not val:
        del inv[ore_name]
    return removed


def get_knowledge():
    """Fallback khi Vector DB chưa sẵn sàng (Oniichan giữ để tương thích)."""
    if os.path.exists('knowledge.json'):
        try:
            with open('knowledge.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                def extract_text(obj):
                    if isinstance(obj, dict):
                        return " ".join([extract_text(v) for v in obj.values()])
                    elif isinstance(obj, list):
                        return " ".join([extract_text(i) for i in obj])
                    return str(obj)
                clean_text = extract_text(data)
                return clean_text[:500]
        except Exception as e:
            print(f"❌ Lỗi đọc JSON: {e}")
            return "Không thể nạp kiến thức bổ sung."
    return "Không tìm thấy file kiến thức."


# --- RAG Nội Sinh: FAISS + sentence-transformers (Oniichan) ---
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _extract_text_from_obj(obj):
    """Bóc toàn bộ text từ object (dict/list/str) trong knowledge.json."""
    if isinstance(obj, dict):
        return " ".join([_extract_text_from_obj(v) for v in obj.values()])
    if isinstance(obj, list):
        return " ".join([_extract_text_from_obj(i) for i in obj])
    return str(obj).strip()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chia text thành các đoạn nhỏ (Oniichan dùng để đưa vào FAISS)."""
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def initialize_vector_db():
    """
    Đọc server_context và system_rules từ knowledge.json, chunk hóa, encode bằng
    all-MiniLM-L6-v2 và tạo index FAISS. Chỉ load model 1 lần (tiết kiệm RAM cho Oniichan).
    """
    global _embedding_model, _faiss_index, _chunk_texts, _vector_db_ready
    if not os.path.exists("knowledge.json"):
        print("⚠️ Oniichan ơi, không tìm thấy knowledge.json — Vector DB bỏ qua.")
        return
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # Chỉ lấy server_context và system_rules như Oniichan yêu cầu
        parts = []
        for key in ("server_context", "system_rules", "rag_dieu_huong"):
            if key in data and data[key]:
                parts.append(_extract_text_from_obj(data[key]))
        full_text = " ".join(parts).strip()
        if not full_text:
            print("⚠️ server_context/system_rules rỗng — Vector DB bỏ qua.")
            return
        # Chia nhỏ thành các đoạn
        _chunk_texts = _chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not _chunk_texts:
            return
        # Load model embedding 1 lần duy nhất (tiết kiệm RAM)
        if _embedding_model is None:
            print("📦 Đang load model embedding (all-MiniLM-L6-v2) lần đầu...")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = _embedding_model.encode(_chunk_texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # Chuẩn hóa để dùng cosine similarity (IndexFlatIP)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        _faiss_index = index
        # Giải phóng bộ nhớ: không giữ bản copy embeddings (FAISS đã lưu), Oniichan
        del embeddings
        _vector_db_ready = True
        print(f"✅ Vector DB (FAISS) sẵn sàng — {len(_chunk_texts)} chunks.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Vector DB: {e}")
        _vector_db_ready = False


def search_vector_db(query: str) -> str:
    """
    Tìm các đoạn thông tin liên quan nhất tới câu hỏi của Oniichan, trả về 1 chuỗi.
    Nếu Vector DB chưa sẵn sàng thì fallback get_knowledge().
    """
    global _embedding_model, _faiss_index, _chunk_texts, _vector_db_ready
    if not _vector_db_ready or _faiss_index is None or not _chunk_texts:
        return get_knowledge()
    try:
        q_emb = _embedding_model.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)
        k = min(TOP_K_RETRIEVAL, len(_chunk_texts))
        scores, indices = _faiss_index.search(q_emb, k)
        lines = []
        for idx in indices[0]:
            if 0 <= idx < len(_chunk_texts):
                lines.append(_chunk_texts[idx])
        return "\n".join(lines).strip() if lines else get_knowledge()
    except Exception as e:
        print(f"❌ Lỗi search Vector DB: {e}")
        return get_knowledge()


# --- RAG Trên Mạng: DuckDuckGo (Oniichan hóng tin 2026) ---
WEB_SEARCH_KEYWORDS = (
    "thời tiết", "tin tức", "giá ", "giá cả", "weather", "news", "2026",
    "mới nhất", "hôm nay", "hiện tại", "giá vàng", "tỷ giá", "giá coin",
    "crypto", "bitcoin", "hôm nay thế nào", "cập nhật", "đang diễn ra"
)


# --- Log chat lên server để debug (Oniichan) ---
CHAT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs")
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

def log_chat_to_server(user_name: str, user_id: str, question: str, answer: str, guild_name: str = ""):
    """Ghi log mỗi lần AI chat lên file trên server host để debug."""
    try:
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        log_file = os.path.join(CHAT_LOG_DIR, f"chat_{date_str}.log")
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] user={user_name}({user_id}) guild={guild_name} | Q: {question[:200]}{'...' if len(question)>200 else ''} | A: {answer[:300]}{'...' if len(answer)>300 else ''}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"❌ Lỗi ghi chat log: {e}")


def need_web_search(question: str) -> bool:
    """Oniichan ơi: phát hiện câu hỏi cần thông tin thực tế mới nhất (thời tiết, tin tức, giá...)."""
    q = question.lower().strip()
    return any(kw in q for kw in WEB_SEARCH_KEYWORDS)


def search_web(query: str, max_results: int = 5) -> str:
    """
    Tìm kiếm web bằng DuckDuckGo, trả về chuỗi tóm tắt kết quả cho Oniichan.
    Nếu không có thư viện hoặc lỗi thì trả về chuỗi rỗng.
    """
    if not HAS_DUCKDUCKGO:
        return ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        parts = []
        for r in results[:max_results]:
            title = r.get("title", "")
            body = r.get("body", "")
            if title or body:
                parts.append(f"- {title}: {body[:200]}" + ("..." if len(body) > 200 else ""))
        return "\n".join(parts).strip() if parts else ""
    except Exception as e:
        print(f"❌ Lỗi DuckDuckGo search: {e}")
        return ""


# ============ AI CHAT COMMANDS ============



ore = {
    'dirt': 1000,
    'stone': 4000,
    'calcite': 15,
    'bauxile':12,
    'iron': 10,
    'copper': 5,
    'sliver': 3,
    'gold_ore': 1,
    'coal': 200,
    'quartz': 25,
    'feldspar': 20,
    'gypsum': 18,
    'halite': 15,
    'fluorite': 12,
    'apatite': 10,
    'magnetite': 8,
    'hematite': 7,
    'galena': 6,
    'sphalerite': 5,
    'chalcopyrite': 4,
    'cassiterite': 4,
    'bauxite': 3,
    'ilmenite': 3,
    'rutile': 3,
    'molybdenite': 2,
    'cinnabar': 2,
    'pyrite': 2,
    'talc': 2,
    'graphite': 2,
    'chromite': 2,
    'uraninite': 1,
    'pitchblende': 1,
    'columbite': 1,
    'tantalite': 1,
    'wolframite': 1,
    'scheelite': 1,
    'zircon': 1,
    'barite': 1,
    'spodumene': 1,
    'lepidolite': 1,
    'beryl': 1,
    'tourmaline': 1,
    'corundum': 1,
    'diamond': 1
}

price={
    'dirt': 1,
    'stone': 2,
    'calcite': 5,
    'coal': 4,
    'bauxile':8,
    'iron': 10,
    'copper': 15,
    'silver': 30,
    'gold': 50,
    'gold_ore':12,
    'quartz': 6,
    'feldspar': 5,
    'gypsum': 4,
    'halite': 3,
    'fluorite': 8,
    'apatite': 7,
    'magnetite': 12,
    'hematite': 10,
    'galena': 15,
    'sphalerite': 14,
    'chalcopyrite': 16,
    'cassiterite': 20,
    'bauxite': 8,
    'ilmenite': 18,
    'rutile': 22,
    'molybdenite': 25,
    'cinnabar': 28,
    'pyrite': 6,
    'talc': 3,
    'graphite': 5,
    'chromite': 20,
    'uraninite': 40,
    'pitchblende': 42,
    'columbite': 35,
    'tantalite': 38,
    'wolframite': 30,
    'scheelite': 28,
    'zircon': 25,
    'barite': 6,
    'spodumene': 32,
    'lepidolite': 30,
    'beryl': 35,
    'tourmaline': 40,
    'corundum': 45,
    'diamond': 100
}
quantity = {   
    'dirt': 100,
    'stone': 50,
    'calcite': 20,
    'bauxile': 1,
    'iron': 1,
    'copper': 1,
    'silver': 1,
    'gold': 1,
    'quartz': 5,
    'feldspar': 5,
    'gypsum': 4,
    'halite': 4,
    'fluorite': 3,
    'apatite': 3,
    'magnetite': 2,
    'hematite': 2,
    'galena': 2,
    'sphalerite': 2,
    'chalcopyrite': 2,
    'cassiterite': 2,
    'bauxite': 2,
    'ilmenite': 2,
    'rutile': 2,
    'molybdenite': 1,
    'cinnabar': 1,
    'pyrite': 3,
    'talc': 4,
    'graphite': 3,
    'chromite': 1,
    'uraninite': 1,
    'pitchblende': 1,
    'columbite': 1,
    'tantalite': 1,
    'wolframite': 1,
    'scheelite': 1,
    'zircon': 1,
    'barite': 3,
    'spodumene': 1,
    'lepidolite': 1,
    'beryl': 1,
    'tourmaline': 1,
    'corundum': 1,
    'diamond': 1
}
emoji_icon = { 
    'dirt' : '<:dirt:1425134897282027533>',
    'stone': '<:stoness:1425135738134990869>',
    'calcite': '<:calcite:1425860868041736382>',
    'bauxile':'<:bauxile:1425860810843885618>',
    'iron': '<:iron:1425860778124382400>',
    'copper': '🧱',
    'sliver':'<:iron:1425860778124382400>',
    'gold': '<:gold:1426776771335815220>',
    'quartz':'<:Quartz:1425864050037887106>' ,
    'feldspar': '<:feldspar:1426776743158743060>',
    'gypsum': '<:calcite:1425860868041736382>',
    'halite': '<:halite:1426776827044696115>',
    'fluorite':'<:fluorite:1425869841927245914>' ,
    'apatite': '<:Apatite:1426778037390934058>',
    'magnetite':'<:Magnetite:1425869000776220763>',
    'hematite':'<:hematite:1425860840841547898>',
    'galena': '<:galena:1426816156655943701>',
    'sphalerite': '<:sphalerite:1426816178596216944>',
    'chalcopyrite': '<:Chalcopyrite:1425868680495108137>',
    'cassiterite': '<:Cassiterite:1426813557533573140>',
    'bauxite': '<:bauxile:1425860810843885618>',
    'ilmenite':'<:Ilmenite:1425869295681802372>' ,
    'rutile':'<:rutite:1426817592194236506>',
    'molybdenite': '<:Molybdenite:1426817671097356369>',
    'cinnabar':'<:cinnabar:1434922309554012180>' ,
    'pyrite': '<:pyrite:1426817047760998481>',
    'talc': '<:talc:1434922300997636136>',
    'graphite':'<:graphite:1434560758628225206>' ,
    'coal':'<:coal:1434560760930893845>',
    'chromite': '',
    'uraninite': '<:uranite:1434922298795888791>',
    'pitchblende':'<:uranite:1434922298795888791> ',
    'columbite':'' ,
    'tantalite': '',
    'wolframite':'' ,
    'scheelite':'' ,
    'zircon': '',
    'barite': '',
    'spodumene': '',
    'lepidolite':'' ,
    'beryl': '<:beryl:1434922296505532496>',
    'tourmaline': '',
    'corundum':'' ,
    'diamond': '💎',
    'iron_ingot': '<:iron_ingot:1434922305703907492>',
    'copper_ingot': '<:copper_ingot:1434922303782785157>',
    'gold_ingot': '<:gold_bar:1434922294261715034>',
    'silver_ingot': '🥈',
    'stone_brick': '<:brick:1427000000000000000>',
    'clay': '🧱',
    'slag': '<:slag:1434922307846934579>',
}
PICKAXES = {
    'default_pickaxe': {'min_multiplier': 1.0, 'max_multiplier': 1.0, 'emoji': '🪨','price': 0},
    'wood_pickaxe': {'min_multiplier': 1.1, 'max_multiplier': 1.5, 'emoji': '<:Wooden_Pickaxe:1434556115655594126>','price': 10000},
    'stone_pickaxe': {'min_multiplier': 1.5, 'max_multiplier': 2.3, 'emoji': '<:Stone_Pickaxe:1434556117773451365>','price': 50000},
    'iron_pickaxe': {'min_multiplier': 2.4, 'max_multiplier': 3.6, 'emoji': '<:Iron_Pickaxe:1434556113147269302>','price': 150000},
    'gold_pickaxe': {'min_multiplier': 3.7, 'max_multiplier': 4.2, 'emoji': '<:Golden_Pickaxe:1434556111209631917>','price': 2000000},
    'diamond_pickaxe': {'min_multiplier': 4.8, 'max_multiplier': 6.9, 'emoji': '<:Diamond_Pickaxe:1434556109053497486>','price': 15000000},


}
# Thêm emoji cho pickaxe vào emoji_icon
emoji_icon.update({
    'default_pickaxe': '🪨',
    'wood_pickaxe': '<:Wooden_Pickaxe:1434556115655594126>',
    'stone_pickaxe': '<:Stone_Pickaxe:1434556117773451365>',
    'iron_pickaxe': '<:Iron_Pickaxe:1434556113147269302>',
    'gold_pickaxe': '<:Golden_Pickaxe:1434556111209631917>',
    'diamond_pickaxe': '<:Diamond_Pickaxe:1434556109053497486>'
})
PICKAXE_ALIASES = {
    'wood_pickaxe': 'cupgo',
    'stone_pickaxe': 'cupda',
    'iron_pickaxe': 'cupsat',
    'gold_pickaxe': 'cupvang',
    'diamond_pickaxe': 'cupkimcuong',
    # Nếu bạn dùng alias khác, thêm vào đây
}




# prices for smelted/processed items
price.update({
    'iron_ingot': 25,
    'copper_ingot': 20,
    'gold_ingot': 80,
    'silver_ingot': 40,
    'stone_brick': 5,
    'clay': 2,
    'slag': 0
})

# === CÂY TRỒNG ===
crops = {
    'wheat':     {'grow_time': 180, 'base_yield': 3, 'price': 15, 'emoji': '🌾'},
    'carrot':    {'grow_time': 240, 'base_yield': 2, 'price': 25, 'emoji': '🥕'},
    'potato':    {'grow_time': 300, 'base_yield': 4, 'price': 20, 'emoji': '🥔'},
    'tomato':    {'grow_time': 360, 'base_yield': 3, 'price': 35, 'emoji': '🍅'},
    'corn':      {'grow_time': 420, 'base_yield': 2, 'price': 50, 'emoji': '🌽'},
}




# Giá bán nông sản
price.update({name: data['price'] for name, data in crops.items()})

# Emoji
emoji_icon.update({name: data['emoji'] for name, data in crops.items()})


intents = discord.Intents.default()
intents.message_content = True  
intents.members = True
intents.presences = True

bot = commands.Bot(command_prefix=['p','P'], intents=intents)
bot.remove_command('help')
channel_ID = 1412439224342548540
auto_send = False
auto_daily= False
WELCOME_CHANNEL_ID = 1297128688801808434

@bot.event
async def on_ready():
    print(f'Bot đã đăng nhập với tên: {bot.user}')
    try:
        await bot.tree.sync()
        print("App commands synced.")
    except Exception:
        pass
    
    try:
        await bot.load_extension('commands') 
        print("Đã tải extension 'commands' thành công.")
    except Exception as e:
        print(f"LỖI: Không thể tải extension 'commands': {e}")
        
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    raise error

@bot.event
async def on_member_join(member):

    
    
    if not WELCOME_CHANNEL_ID:
        print("Cảnh báo: WELCOME_CHANNEL_ID chưa được thiết lập!")
        return 
        
   
    channel = member.guild.get_channel(WELCOME_CHANNEL_ID)
    
    
    if channel is None:
        print(f"LỖI: Không tìm thấy kênh chào mừng với ID: {WELCOME_CHANNEL_ID}")
        return

    embed = discord.Embed(
        title=f"🎉 Chào Mừng {member.name}!",
        description=f"Cảm ơn cậu đã tham gia sv**{member.guild.name}**!\nHãy đọc <#1303933198912192532> và tải KVGDPS tại <#1291732977595973792> nhé!",
        color=discord.Color.from_rgb(102, 204, 255), # Màu xanh nước biển nhạt
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    

    embed.add_field(
        name="cậu là thành viên thứ:", 
        value=f"**{len(member.guild.members):,}**", 
        inline=True
    )

    # Thêm ảnh đại diện của thành viên
    if member.avatar:
        embed.set_thumbnail(url=member.avatar.url)
    

    embed.set_footer(
        text=f"ID người dùng: {member.id}", 
        icon_url=member.guild.icon.url if member.guild.icon else None
    )

    await channel.send(f"Xin chào mừng {member.mention} đã đến với server! <a:tada:>", embed=embed)

    
    
@bot.command(name='ai')
async def chat_ai(ctx, *, question: str):
    """Hỏi AI local bất kỳ câu hỏi nào — RAG nội sinh (FAISS) + web search (Oniichan)."""
    async with ctx.typing():
        try:
            # RAG Nội Sinh: chỉ lấy đoạn thông tin liên quan nhất tới câu hỏi (Oniichan)
            knowledge = search_vector_db(question)
            # RAG Trên Mạng: nếu câu hỏi cần tin mới nhất (thời tiết, tin tức, giá...) thì search web
            web_extra = ""
            if need_web_search(question):
                web_extra_raw = search_web(question, max_results=5)
                if web_extra_raw:
                    web_extra = f"\n\n[Thông tin từ web (Oniichan hóng hớt tin mới):]\n{web_extra_raw[:800]}"
            # Ghép knowledge + web vào system/user để giữ đúng vibe wibu lầy lội
            system_extra = ""
            if knowledge:
                system_extra = f"\n[Kiến thức server/rule liên quan:]\n{knowledge[:600]}"
            if web_extra:
                system_extra += web_extra
            # Hard rule: chỉ anh/oniichan, cấm giả trân / văn mẫu từ chối
            hard_rules = """
[HARD RULE - BẮT BUỘC]
- Tuyệt đối CHỈ xưng anh hoặc oniichan với người dùng. CẤM dùng: bạn, bạn ơi, cậu, quý khách, người dùng, v.v.
- CẤM tone giả trân, văn mẫu corporate: "có vẻ bạn đang muốn", "tôi không được phép", "tôi có thể giúp đỡ bạn", "tôi không được chia sẻ"... Trả lời thẳng, wibu/lầy, không lên mặt đạo đức.
- Nếu chạm đề nhạy cảm: trả lời khách quan, khoa học hoặc lầy lội — không từ chối kiểu văn mẫu."""
            system_msg = f"Mày là một em Waifu AI thông minh, lầy lội, chuyên hỗ trợ code. Trả lời thẳng vào vấn đề bằng ngôn ngữ teen/wibu. Tuyệt đối không được nói 'đây là một vấn đề khó' hay 'mình nên hiểu rõ'. Nếu không biết thì hãy chửi nhẹ anh/oniichan một câu hoặc tỏ ra dỗi thay vì luyên thuyên.{hard_rules}{system_extra}"
            # Llama 3.2 format (Meta)
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            response = llm(prompt, max_tokens=1024, temperature=1.2, top_p=0.9, stop=["<|eot_id|>", "<|end_header_id|>"])
            answer = response['choices'][0]['text'].strip()
            
            if len(answer) > 1900:
                answer = answer[:1900] + "..."
            
            # Log chat lên server host để debug (Oniichan)
            log_chat_to_server(
                user_name=ctx.author.name,
                user_id=str(ctx.author.id),
                question=question,
                answer=answer,
                guild_name=ctx.guild.name if ctx.guild else ""
            )
            
            embed = discord.Embed(title=f"{AI_NAME} 🦙", description=answer, color=discord.Color.green())
            embed.set_footer(text=f"Hỏi bởi {ctx.author.name}")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Lỗi khi gọi AI: {e}")


@bot.command(name='aitest')
async def ai_test(ctx):
    """Kiểm tra xem AI local có hoạt động không"""
    async with ctx.typing():
        try:
            test_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nXin chào, bạn có hoạt động không?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            response = llm(test_prompt, max_tokens=50, temperature=0.7, stop=["<|eot_id|>", "<|end_header_id|>"])
            answer = response['choices'][0]['text'].strip()
            await ctx.send(f"✅ **AI Local đang hoạt động!**\n\nTest response: {answer}")
        except Exception as e:
            await ctx.send(f"❌ AI không hoạt động: {e}")


@bot.command(name='aiinfo')
async def ai_info(ctx):
    """Hiển thị thông tin về AI model"""
    embed = discord.Embed(title=f"🧠 {AI_NAME} — Thông tin AI Local", color=discord.Color.blue())
    embed.add_field(name="Model", value="Llama 3.2 1B Instruct", inline=False)
    embed.add_field(name="Quantization", value="Q4_K_M", inline=True)
    embed.add_field(name="Context", value="2048 tokens", inline=True)
    embed.add_field(name="Threads", value="2", inline=True)
    embed.add_field(name="Path", value=AI_MODEL_PATH, inline=False)
    await ctx.send(embed=embed)
# đầu lệnh mine
@bot.command()
@commands.cooldown(rate=1,per=15,type=commands.BucketType.user)
async def mine(ctx):
    user_id = str(ctx.author.id)
    
    user_data = get_user_data(user_id) 
    inv = user_data['inventory']
    
    # ⛏️ Logic kiểm tra và áp dụng Pickaxe (giữ nguyên)
    best_pickaxe = 'default_pickaxe'
    ownable_pickaxes = {k: v for k, v in PICKAXES.items() if k != 'default_pickaxe'}
    for pickaxe_name, props in ownable_pickaxes.items():
        if get_total_ore_count(inv, pickaxe_name) > 0:
            current_best_multiplier = PICKAXES[best_pickaxe]['max_multiplier']
            if props['max_multiplier'] > current_best_multiplier:
                best_pickaxe = pickaxe_name
                
    # Lấy thuộc tính của cuốc (DÙNG CHO CẢ SỐ LƯỢNG VÀ ĐỘ HIẾM)
    props = PICKAXES[best_pickaxe]
    
    # 1. Tính SỐ LƯỢNG (Multiplier)
    best_multiplier = (props['min_multiplier'], props['max_multiplier'])
    multiplier = random.uniform(best_multiplier[0], best_multiplier[1])
    pickaxe_emoji = emoji_icon.get(best_pickaxe, "❓")
    
    if best_pickaxe == 'default_pickaxe':
        pickaxe_msg = " (dùng xẻng)"
    else:
        pickaxe_msg = f" (dùng {pickaxe_emoji} **{best_pickaxe}**)"

    # *** LOGIC MỚI: TẠO TRỌNG SỐ ĐỘNG DỰA TRÊN CỐC ĐÀO ***
    rarity_reduction_factor = props['min_multiplier'] 
    
    dynamic_weights = {}
    for ore_name, original_weight in ore.items():
        if 0 < original_weight < 30:
            # Quặng hiếm (weight < 30) sẽ phổ biến hơn khi dùng cuốc xịn
            new_weight = max(1, int(original_weight * rarity_reduction_factor))
            dynamic_weights[ore_name] = new_weight
        else:
            # Giữ nguyên trọng số cho quặng phổ biến (dirt, stone)
            dynamic_weights[ore_name] = original_weight

    # *** Dùng `dynamic_weights` để tìm quặng ***
    found_ore = random.choices(list(dynamic_weights.keys()), weights=list(dynamic_weights.values()), k=1)[0]
    
    # Tính số lượng quặng (giữ nguyên)
    base_amount = random.randint(1, 10)
    final_amount = int(base_amount * multiplier) # Multiplier từ cuốc
    
    if final_amount == 0:
        final_amount = 1
        
    emoji= emoji_icon.get(found_ore, "")
    
    # *** LOGIC MỚI: TÍNH TOÁN CHẤT LƯỢNG DỰA TRÊN LOẠI QUẶNG VÀ CÚP ĐÀO ***
    original_weight_check = ore.get(found_ore, 0) # Lấy trọng số GỐC
    
    if found_ore in ('dirt', 'stone'):
        # Yêu cầu 1: dirt và stone chất lượng cố định là 100%
        quality_percent = 100
        quality_msg = "" # Yêu cầu 2: Không hiển thị chất lượng cho dirt/stone
    elif 0 < original_weight_check < 30:
        # Quặng hiếm (original weight < 30)
        if best_pickaxe == 'default_pickaxe':
            # Dùng default_pickaxe, chất lượng 5% - 75%
            quality_percent = random.randint(5, 75) 
        else:
            # Dùng cuốc xịn, chất lượng cao hơn
            quality_percent = random.randint(75, 100) 
        quality_msg = f" ({quality_percent}% chất lượng)"
    else:
        # Quặng phổ biến khác
        quality_percent = random.randint(50, 100)
        quality_msg = f" ({quality_percent}% chất lượng)"
    
    # Bạn cần đảm bảo hàm add_ore_with_quality tồn tại và hoạt động
    add_ore_with_quality(inv, found_ore, quality_percent=quality_percent, qty=final_amount)
    
    # save data 🐿️🥛
    save_data(player_inventory)

    # Embed đẹp cho lệnh !mine
    ore_display = f"{emoji} **{found_ore}**" if emoji else f"**{found_ore}**"
    quality_display = f"{quality_percent}%" if found_ore not in ('dirt', 'stone') else "—"
    pickaxe_display = f"{pickaxe_emoji} {best_pickaxe}" if best_pickaxe != 'default_pickaxe' else "🪓 Xẻng mặc định"

    embed = discord.Embed(
        title="⛏️ Đào quặng thành công",
        description=f"{ctx.author.mention} vừa đào được **{final_amount}** {ore_display}!",
        color=discord.Color.from_rgb(139, 90, 43),  # màu nâu đất
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )
    embed.add_field(name="📦 Số lượng", value=f"**{final_amount}**", inline=True)
    embed.add_field(name="🪨 Loại quặng", value=ore_display, inline=True)
    embed.add_field(name="✨ Chất lượng", value=quality_display, inline=True)
    embed.add_field(name="🪓 Cuốc sử dụng", value=pickaxe_display, inline=False)
    if ctx.author.avatar:
        embed.set_thumbnail(url=ctx.author.avatar.url)
    embed.set_footer(text=f"Mine bởi {ctx.author.name} • Cooldown 15s", icon_url=ctx.guild.icon.url if ctx.guild and ctx.guild.icon else None)
    await ctx.send(embed=embed)


@mine.error
async def mine_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        seconds = error.retry_after
        await ctx.send(f"⏳ **{ctx.author.mention}**, từ từ bạn ơi, đợi **{seconds:.1f} giây** nữa.")
    
    
    elif isinstance(error, commands.CommandInvokeError):  
        await ctx.send(f"❌ Đã xảy ra lỗi nội bộ khi thực thi lệnh `mine`: `{error.original}`. Vui lòng báo cáo lỗi này.")
    
    
    else:
        await ctx.send(f"❌ Đã xảy ra lỗi không xác định: `{error}`. Vui lòng báo cáo lỗi này.")
# cuối lệnh mine

# BẮT ĐẦU LỆNH SHOP MỚI TẠI ĐÂY

@bot.group(invoke_without_command=True)
async def shop(ctx):
    """Hiển thị cửa hàng, dùng 'shop buy [tên_mặt_hàng]' để mua."""
    
    shop_embed = discord.Embed(
        title="🛍️ Cửa Hàng Cuốc Đào (Pickaxe Shop)",
        description="Dùng cuốc xịn hơn để đào được nhiều quặng hơn và giảm độ hiếm của quặng hiếm!",
        color=discord.Color.gold()
    )

    item_list = ""
    
    # Bỏ qua 'default_pickaxe' vì nó không bán
    for name, props in PICKAXES.items():
        if name == 'default_pickaxe':
            continue
        
        emoji = emoji_icon.get(name, "❓")
        price_str = f"{props['price']:,} {KESLING_ICON}" 
        multiplier_str = f"x{props['min_multiplier']} - x{props['max_multiplier']} Quặng"
        
        # Lấy tên tiếng Việt không dấu đầu tiên từ alias để làm ví dụ
        alias_name = next((vn_name for vn_name, key in PICKAXE_ALIASES.items() if key == name), name)

        item_list += f"**{emoji} {name.upper()}**\n"
        item_list += f"> **Giá:** {price_str}\n"
        item_list += f"> **Hiệu suất:** {multiplier_str}\n"
        item_list += f"> **Cách mua:** `shop buy {alias_name}`\n\n"

    shop_embed.add_field(name="⛏️ Danh Sách Cuốc Đào", value=item_list, inline=False)
    shop_embed.set_footer(text="Gợi ý: Dùng cú pháp tiếng Việt không dấu: shop buy cuocgo")
    
    await ctx.send(embed=shop_embed)

@shop.command(name="buy")
async def shop_buy(ctx, item_name: str = None):
    """Mua một vật phẩm từ cửa hàng."""
    user_id = str(ctx.author.id)
    user_data = get_user_data(user_id)
    
    if item_name is None:
        return await ctx.send("❌ Vui lòng cho biết bạn muốn mua gì. Ví dụ: `shop buy cuocgo`")

    # Xử lý input: đưa về chữ thường và loại bỏ khoảng trắng
    input_key = item_name.lower().replace(" ", "")
    
    # --- BẮT ĐẦU FIX LỖI TÊN VẬT PHẨM (ALIAS) ---
    # 1. Tạo map ngược: {alias: key} để tìm tên chính thức từ tên viết tắt/tiếng Việt
    REVERSE_ALIASES = {v: k for k, v in PICKAXE_ALIASES.items()}
    
    # 2. Ánh xạ input_key sang item_key (tên chính thức trong PICKAXES)
    # Nếu người dùng nhập alias (ví dụ: 'cupgo'), item_key sẽ là 'wood_pickaxe'
    if input_key in REVERSE_ALIASES:
        item_key = REVERSE_ALIASES[input_key] 
    # Nếu người dùng nhập tên chính thức (ví dụ: 'wood_pickaxe'), item_key vẫn giữ nguyên
    else:
        item_key = input_key
    # --- KẾT THÚC FIX LỖI TÊN VẬT PHẨM ---
    
    # Kiểm tra
    if item_key not in PICKAXES or item_key == 'default_pickaxe':
        return await ctx.send(f"❌ Không tìm thấy vật phẩm **{item_name}** trong cửa hàng Cuốc Đào. Vui lòng kiểm tra lại tên vật phẩm.")

    item_props = PICKAXES[item_key]
    price = item_props['price']
    
    if user_data.get('money', 0) < price:
        return await ctx.send(f"💰 Bạn không đủ tiền! Bạn cần **{price:,} {KESLING_ICON}s** để mua {item_key}.")
        
    # --- Logic Mua hàng ---
    
    # 1. Trừ tiền (ĐÃ SỬA LỖI NAMEERROR TỪ LẦN TRƯỚC)
    user_data['money'] = user_data.get('money', 0) - price
    
    # 2. Thêm Pickaxe vào kho đồ (inventory)
    inv = user_data['inventory']
    inv[item_key] = inv.get(item_key, 0) + 1
    
    save_data(player_inventory) 
    
    emoji = emoji_icon.get(item_key, "❓")
    await ctx.send(f"✅ Chúc mừng, {ctx.author.mention}! Bạn đã mua thành công **{emoji} {item_key}** với giá **{price:,} {KESLING_ICON}s**! Giờ bạn có thể dùng nó để đào.")
# -----------------------------------------------------------------------------------------------------------------------#

@bot.command(name="ping")
async def ping(ctx):
    # --- GIAI ĐOẠN 1: HIỆN THÔNG SỐ CƠ BẢN NGAY LẬP TỨC ---
    ws_latency = round(bot.latency * 1000)
    start_time = time.time()
    
    embed = discord.Embed(
        title="🛰️ KATVIET SYSTEM STATUS",
        color=0x5865F2,
        description="📡 *Đang kiểm tra các thông số kỹ thuật...*"
    )
    embed.add_field(name="📶 Server Latency", value=f"`{ws_latency}ms`", inline=True)
    embed.add_field(name="⚡ API Latency", value="`Calculating...`", inline=True)
    embed.add_field(name="🤖 AI System", value="`Scanning List...`", inline=False)
    
    message = await ctx.reply(embed=embed)
    
    # Tính API Latency sau khi gửi tin nhắn xong
    api_latency = round((time.time() - start_time) * 1000)

    # --- GIAI ĐOẠN 2: DÒ LIST AI ---
    online_models = []
    active_model = "None"
    
    # Cập nhật embed lần 1 để hiện API Latency trước cho người dùng yên tâm
    embed.set_field_at(1, name="⚡ API Latency", value=f"`{api_latency}ms`", inline=True)
    await message.edit(embed=embed)

    # Bắt đầu dò list AI
    for model_name in models_to_try:
        try:
            # Gửi test siêu ngắn với timeout thấp (2s) để nhanh hơn
            test = await g4f.ChatCompletion.create_async(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                timeout=2 
            )
            if test:
                online_models.append(model_name)
                if active_model == "None":
                    active_model = model_name
        except:
            continue

    # --- GIAI ĐOẠN 3: HOÀN TẤT DASHBOARD ---
    ai_status = "🟢 Online" if active_model != "None" else "🔴 All Offline"
    final_color = 0x2ecc71 if active_model != "None" else 0xe74c3c
    
    final_embed = discord.Embed(title="🛰️ KATVIET SYSTEM STATUS", color=final_color)
    final_embed.add_field(name="📶 Server Latency", value=f"`{ws_latency}ms`", inline=True)
    final_embed.add_field(name="⚡ API Latency", value=f"`{api_latency}ms`", inline=True)
    final_embed.add_field(name="🤖 AI System", value=f"`{ai_status}`", inline=True)
    
    final_embed.add_field(name="🧠 Active Model", value=f"**{active_model}**", inline=False)
    
    if online_models:
        final_embed.add_field(
            name="✅ Available Models", 
            value=f"```fix\n" + "\n".join([f"• {m}" for m in online_models]) + "```", 
            inline=False
        )

    final_embed.set_footer(text=f"Check by {ctx.author.name} | Vibe Coding by Ming Seng", icon_url=ctx.author.avatar.url)
    final_embed.timestamp = datetime.datetime.now()

    await message.edit(embed=final_embed)
@bot.command()
async def cf(ctx, guess: str = None, bet: int = None):
    user_id = str(ctx.author.id)
    if guess is None or bet is None:
        await ctx.send(f"{ctx.author.mention} dùng: p {KESLING_ICON}flip <h/t> hoặc <ngua/sap> <số_tiền_cược>")
        return
        
    guess = guess.lower()
    if guess not in ["h", "t","ngua","sap"]:
        await ctx.send(f"{ctx.author.mention} chỉ được đoán 'h <ngửa>' hoặc 't <sấp>'.")
        return
        
    if bet <= 0:
        await ctx.send(f"{ctx.author.mention} số tiền cược phải lớn hơn 0.")
        return
        
    user_data = get_user_data(user_id)
    
    if user_data['money'] < bet:
        await ctx.send(f"{ctx.author.mention} bạn không đủ tiền để cược.")
        return
        
    result = random.choice(["h","t","ngua","sap"])
    if result == "h":
        result = "ngua"
    else:
        result = "sap"

    if guess in ["h", "ngua"]:
        user_guess = "ngua"
    else:
        user_guess = "sap"
        
    if user_guess == result:
        user_data['money'] += bet 
        await ctx.send(f"{ctx.author.mention} Đúng rồi! Kết quả là {result}. Bạn nhận được {bet*2} {KESLING_ICON}s!")
    else:
        user_data['money'] -= bet
        await ctx.send(f"{ctx.author.mention} Sai rồi! Kết quả là {result}. Bạn mất {bet} {KESLING_ICON}s!")
        
    # save data 🐸
    save_data(player_inventory) 

# --- CÁC HÀM HỖ TRỢ PHÂN TRANG (BUTTON) ---
# Yêu cầu import discord, math, View, Button, button, ButtonStyle
def create_inventory_pages(title: str, lines: list, per_page: int, color=discord.Color.blue()):
    """Tạo danh sách Embed từ các dòng nội dung, chia trang theo per_page."""
    pages = []
    # Tính tổng số trang
    total_pages = math.ceil(len(lines) / per_page) 
    
    if total_pages == 0:
        # Trường hợp không có vật phẩm nào, vẫn trả về danh sách rỗng
        return pages

    for i in range(total_pages):
        # Cắt nội dung cho từng trang
        start = i * per_page
        end = (i + 1) * per_page
        page_lines = lines[start:end]

        # Tạo Embed cho trang hiện tại
        embed = discord.Embed(
            title=f"{title} (Trang {i + 1}/{total_pages})",
            description="\n".join(page_lines),
            color=color
        )
        pages.append(embed)
        
    return pages

class InventoryPaginationView(View):
    """View quản lý các nút chuyển trang cho kho đồ."""
    def __init__(self, pages: list[discord.Embed], author_id: int):
        super().__init__(timeout=180) # Timeout sau 3 phút
        self.pages = pages
        self.current_page = 0
        self.message = None
        self.author_id = author_id 
        self.update_buttons()

    async def on_timeout(self):
        # Xóa các nút khi hết thời gian
        if self.message:
            for item in self.children:
                item.disabled = True
            await self.message.edit(view=self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # Chỉ cho phép người dùng đã gọi lệnh tương tác
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("❌ Bạn không phải là người đã gọi lệnh này.", ephemeral=True)
            return False
        return True

    def update_buttons(self):
        """Cập nhật trạng thái bật/tắt của các nút."""
        # Nút 'Trang Trước' (prev_button) - index 0
        self.children[0].disabled = self.current_page == 0
        # Nút 'Trang Sau' (next_button) - index 1
        self.children[1].disabled = self.current_page == len(self.pages) - 1

    @button(label="Trang Trước", style=ButtonStyle.blurple, emoji="⬅️")
    async def prev_button(self, interaction: discord.Interaction, button: Button):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_buttons()
            # Cập nhật tin nhắn với trang mới
            await interaction.response.edit_message(embed=self.pages[self.current_page], view=self)

    @button(label="Trang Sau", style=ButtonStyle.blurple, emoji="➡️")
    async def next_button(self, interaction: discord.Interaction, button: Button):
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
            self.update_buttons()
            # Cập nhật tin nhắn với trang mới
            await interaction.response.edit_message(embed=self.pages[self.current_page], view=self)

    @button(label="Kết Thúc", style=ButtonStyle.red, emoji="🛑")
    async def stop_button(self, interaction: discord.Interaction, button: Button):
        # Tắt tất cả các nút và ngắt View
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(view=self)
        self.stop()
        
async def send_paginated_via_ctx(ctx: commands.Context, title: str, lines: list, per_page: int):
    """Hàm gửi nội dung phân trang có nút bấm."""
    if not lines:
        return # Không có nội dung để phân trang
        
    pages = create_inventory_pages(title, lines, per_page, color=discord.Color.blue())
    
    if not pages:
        return

    if len(pages) == 1:
        # Nếu chỉ có 1 trang, gửi Embed mà không cần View (nút)
        await ctx.send(embed=pages[0])
        return

    # Nếu có nhiều trang, khởi tạo View và gửi tin nhắn
    view = InventoryPaginationView(pages, ctx.author.id)
    # Lần gửi đầu tiên
    message = await ctx.send(embed=pages[0], view=view)
    view.message = message # Lưu lại tin nhắn để View có thể chỉnh sửa nó
    
    return message
# --- KẾT THÚC CÁC HÀM HỖ TRỢ PHÂN TRANG (BUTTON) ---

# SỬA LỆNH BAG (TÚI ĐỒ) - Sử dụng hàm phân trang mới có nút bấm
@bot.command(name="bag")
async def bag(ctx, member: discord.Member = None):
    """Xem túi: p bag (xem của bạn) hoặc p bag @user (xem của người được mention). Hỗ trợ phân trang bằng nút bấm để tránh lỗi 400."""
    target = member or ctx.author
    user_id = str(target.id)
    user_data = get_user_data(user_id) # Thay bằng hàm get_user_data thực tế của bạn

    # Giả sử bạn đã định nghĩa: 
    # - KESLING_ICON (Icon tiền tệ)
    # - emoji_icon (Dictionary mapping tên quặng -> emoji)
    # - get_user_data (Hàm tải data người dùng)
    
    inv = user_data.get('inventory', {})
    money = user_data.get('money', 0)
    
    # 1. Gửi thông tin tổng quan và tiền tệ (Embed 1)
    money_embed = discord.Embed(
        title=f"🎒 Kho của {target.display_name}",
        description=f"💰 Tiền: **{money:,} {KESLING_ICON}**\n\n--- Danh Sách Tài Nguyên ---",
        color=discord.Color.gold()
    )
    await ctx.send(embed=money_embed)

    if not inv:
        await ctx.send("Kho tài nguyên trống rỗng.")
        return

    ore_lines = []
    
    # Sắp xếp theo tên quặng (A-Z) cho dễ tra cứu
    sorted_inv_items = sorted(inv.items(), key=lambda item: item[0]) 

    for ore_name, amount in sorted_inv_items:
        # Giả định emoji_icon tồn tại (ví dụ: emoji_icon = {"gold_ore": "🪙", ...})
        emoji = emoji_icon.get(ore_name, "") 
        # Định dạng tên quặng (ví dụ: 'gold_ore' thành 'Gold ore')
        display_name = ore_name.replace('_', ' ').capitalize()
        
        if isinstance(amount, dict):
            # Xử lý quặng có chất lượng (dict)
            total_count = sum(int(v) for v in amount.values())
            # parts: [f"{cnt}x{q}%"] - Sắp xếp chất lượng từ cao xuống thấp
            parts = [f"{cnt}x{q}%" for q, cnt in sorted(amount.items(), key=lambda x: int(x[0]), reverse=True)]
            
            line = f"{emoji} **{display_name}**\n> **Tổng**: {total_count:,} - **Chi tiết**: *{', '.join(parts)}*"
        else:
            # Xử lý vật phẩm/quặng không chất lượng (int)
            line = f"{emoji} **{display_name}**\n> **Số lượng**: {amount:,}"
        
        ore_lines.append(line)

    # 2. Dùng hàm phân trang mới để gửi danh sách tài nguyên (Có nút bấm)
    # per_page=7 để mỗi trang hiển thị 7 mục, đảm bảo an toàn cho Discord Embed.
    await send_paginated_via_ctx(
        ctx, 
        f"Chi Tiết Kho Tài Nguyên", 
        ore_lines, 
        per_page=7
    )


@bot.command(name="me")
async def me_command(ctx):
    """Xem số tiền và ticket của bản thân."""
    user_id = str(ctx.author.id)
    user_data = get_user_data(user_id)
    
    money = user_data.get('money', 0)
    tickets = user_data.get('tickets', 0)
    
    response = (
        f"**💸 Ví Tiền của {ctx.author.display_name}**:\n"
        f"💰 Tiền mặt (Money): **{money:,} {KESLING_ICON}s**\n"
        f"🎟️ Vé (Tickets): **{tickets:,}**"
    )
    await ctx.send(response)

# ---------------- LỆNH ACC / USER ----------------
@bot.command(name="acc", aliases=['user'])
async def user_account_command(ctx, member: Optional[discord.Member] = None):
    """Xem số tiền và ticket của người khác. (Dùng lệnh: acc @user hoặc user @user)"""
    
    # Nếu không tag ai, mặc định xem của mình
    if member is None:
        member = ctx.author
        
    user_id = str(member.id)
    user_data = get_user_data(user_id)
    
    money = user_data.get('money', 0)
    tickets = user_data.get('tickets', 0)
    
    response = (
        f"**💸 Ví Tiền của {member.display_name}**:\n"
        f"💰 Tiền mặt (Money): **{money:,} {KESLING_ICON}s**\n"
        f"🎟️ Vé (Tickets): **{tickets:,}**"
    )
    await ctx.send(response)



# LỆNH LUYENKIM ĐÃ CẬP NHẬT
@bot.command(name="luyenkim")
async def luyenkim(ctx, ore_name: str = None, qty: str = "1"):
    """Smelt ores into ingots: p luyenkim (opens form) or p luyenkim <ore> [số_lượng|all]. Cần than (coal) cho kim loại."""
    user_id = str(ctx.author.id)
    user_data = get_user_data(user_id)
    inv = user_data.get('inventory', {})


    smelt_map_data = {
        'iron': ('iron_ingot', 1, 1, True), 
        'magnetite': ('iron_ingot', 1, 1, True),
        'hematite': ('iron_ingot', 1, 1, True), 
        'copper': ('copper_ingot', 1, 1, True), 
        'gold_ore': ('gold_ingot', 1, 1, True), 
        'silver': ('silver_ingot', 1, 1, True), 
        'stone': ('stone_brick', 1, 0, False), 
        'dirt': ('clay', 1, 0, False), # Không cần than
    }
    smelt_map_keys = {k: v[0] for k, v in smelt_map_data.items()}

    async def process_smelt(ctx_or_interaction, user_id, ore_key, amount):
        user_data = get_user_data(user_id)
        inv = user_data.get('inventory', {})
        have = get_total_ore_count(inv, ore_key)
        
        if amount <= 0:
            return "Số lượng phải lớn hơn 0."
        if amount > have:
            return f"Bạn chỉ có {have} {emoji_icon.get(ore_key, '')} {ore_key}."

        out_item, out_qty, coal_cost, needs_coal = smelt_map_data[ore_key]
        
        # --- LOGIC KIỂM TRA THAN ---
        coal_needed = amount * coal_cost
        have_coal = get_total_ore_count(inv, 'coal')
        
        if needs_coal and have_coal < coal_needed:
            return f"🔥 Bạn cần **{coal_needed}** {emoji_icon.get('coal', '')} than để luyện kim {amount} {ore_key}. Bạn chỉ có {have_coal}."

        produced = 0
        slag_item = 'slag'
        slag_count = 0

        # remove units using our quality-aware storage (luôn ưu tiên chất lượng cao nhất)
        removed = remove_ore_units(inv, ore_key, amount, strategy='highest')
        
        # --- LOGIC TIÊU THỤ THAN (nếu có) ---
        if needs_coal and coal_needed > 0:
            inv['coal'] = have_coal - coal_needed
            if inv['coal'] == 0:
                del inv['coal']
        
        # --- LOGIC LÀM CHẢY (SMELT) ---
        for q_str, cnt in removed.items():
            q = int(q_str)
            
            for _ in range(cnt):
                if ore_key in ('dirt', 'stone'):
                    # Đất/đá luôn thành công (100% chance, 100% yield)
                    produced += out_qty
                else:
                    # Công thức luyện kim (Kim loại)
                    # Tính toán tỷ lệ thành công dựa trên Chất lượng Quặng (Quality)
                    # Giả sử Tỷ lệ thành công = 10% + Chất lượng/200.0 (tối đa 60%)
                    success_chance = 0.10 + (q / 200.0) 
                    
                    if random.random() < success_chance:
                        # Thành công: tạo ra thỏi (Ingot)
                        produced += out_qty
                    else:
                        # Thất bại: tạo ra xỉ (Slag)
                        slag_count += 1
                        
        # --- CẬP NHẬT KHO ---
        if produced > 0:
            # Nếu item đã có trong kho, ta coi là không có chất lượng
            inv[out_item] = inv.get(out_item, 0) + produced
        if slag_count > 0:
            inv[slag_item] = inv.get(slag_item, 0) + slag_count

        user_data['inventory'] = inv
        save_data(player_inventory)

        # --- TẠO THÔNG BÁO KẾT QUẢ ĐẸP HƠN ---
        emoji_out = emoji_icon.get(out_item, '✨')
        emoji_slag = emoji_icon.get('slag', '⚒️')
        
        embed = discord.Embed(
            title=f"🔥 Luyện Kim Thành Công {emoji_out}",
            description=f"Đã xử lý **{amount}** {ore_key}.",
            color=discord.Color.green() if produced > 0 else discord.Color.orange()
        )
        
        # Thống kê quặng đã dùng
        ore_used_parts = [f"{cnt}x{q}%" for q, cnt in removed.items()]
        embed.add_field(name=f"Nguyên liệu đã dùng ({ore_key})", value=f"{', '.join(ore_used_parts)}", inline=False)
        
        # Kết quả
        result_str = ""
        if produced:
            result_str += f"**{produced}** {emoji_out} **{out_item}**\n"
        if slag_count:
            result_str += f"**{slag_count}** {emoji_slag} **xỉ** (thất bại)\n"
        if needs_coal:
            result_str += f"**-{coal_needed}** {emoji_icon.get('coal', '⚫')} **than**"
            
        embed.add_field(name="Sản Phẩm và Tiêu Thụ", value=result_str or "Không có sản phẩm nào.", inline=False)

        return embed # Trả về Embed thay vì string

    # If no ore specified, open an interaction form (Select + Modal)
    if ore_name is None:
        # Build a view with a select for ores and a button to input quantity
        class LuyenKimModal(discord.ui.Modal, title="Luyện kim - Nhập số lượng"):
            qty = discord.ui.TextInput(label="Số lượng (hoặc 'all')", placeholder="1", required=True, max_length=10)

            def __init__(self, ore_key, author_id):
                super().__init__()
                self.ore_key = ore_key
                self.author_id = author_id

            async def on_submit(self, interaction: discord.Interaction):
                raw = self.qty.value.strip()
                if raw.lower() == 'all':
                    # compute total available
                    user_data = get_user_data(self.author_id)
                    total = get_total_ore_count(user_data.get('inventory', {}), self.ore_key)
                    amount = total
                else:
                    try:
                        amount = int(raw)
                    except Exception:
                        await interaction.response.send_message("Số lượng không hợp lệ.", ephemeral=True)
                        return
                
                # SỬ DỤNG HÀM PROCESS ĐỂ TRẢ VỀ EMBED
                embed_result = await process_smelt(interaction, self.author_id, self.ore_key, amount)
                
                # Gửi embed (hoặc string nếu là lỗi)
                if isinstance(embed_result, discord.Embed):
                    await interaction.response.send_message(embed=embed_result)
                else:
                    await interaction.response.send_message(embed_result) # Gửi lỗi dưới dạng string
                
        class LuyenKimView(discord.ui.View):
            def __init__(self, author_id):
                super().__init__(timeout=60)
                self.author_id = author_id

            @discord.ui.select(placeholder="Chọn quặng để luyện", min_values=1, max_values=1,
                               options=[discord.SelectOption(label=f"{k} -> {smelt_map_keys[k]}", value=k) for k in smelt_map_keys.keys()])
            async def select_callback(self, interaction: discord.Interaction, select: discord.ui.Select):
                chosen = select.values[0]
                # open modal to input qty
                modal = LuyenKimModal(chosen, interaction.user.id)
                await interaction.response.send_modal(modal)

        view = LuyenKimView(ctx.author.id)
        
        # Cải thiện tin nhắn mở form
        help_embed = discord.Embed(
            title="🏭 Lò Luyện Kim Tự Động",
            description=f"Chọn quặng bạn muốn luyện bên dưới. **Kim loại cần than (coal)** (1 than/1 quặng)!",
            color=discord.Color.blue()
        )
        await ctx.send(embed=help_embed, view=view)
        return

    # else handle direct CLI call
    ore_key = ore_name.lower()
    if ore_key not in smelt_map_keys:
        await ctx.send(f"Không thể luyện kim **{ore_key}**. Các quặng có thể luyện: {', '.join(smelt_map_keys.keys())}")
        return

    # resolve quantity
    if isinstance(qty, str) and qty.lower() == 'all':
        amount = get_total_ore_count(inv, ore_key)
    else:
        try:
            amount = int(qty)
        except Exception:
            await ctx.send("Số lượng không hợp lệ.")
            return

    # SỬ DỤNG HÀM PROCESS ĐỂ TRẢ VỀ EMBED
    embed_result = await process_smelt(ctx, user_id, ore_key, amount)
    
    # Gửi embed (hoặc string nếu là lỗi)
    if isinstance(embed_result, discord.Embed):
        await ctx.send(embed=embed_result)
    else:
        await ctx.send(embed_result) # Gửi lỗi dưới dạng string

# ... (rest of the file)
    # else handle direct CLI call
    ore_key = ore_name.lower()
    if ore_key not in smelt_map:
        await ctx.send(f"Không thể luyện kim {ore_key}. Các quặng có thể luyện: {', '.join(smelt_map.keys())}")
        return

    # resolve quantity
    if isinstance(qty, str) and qty.lower() == 'all':
        amount = get_total_ore_count(inv, ore_key)
    else:
        try:
            amount = int(qty)
        except Exception:
            await ctx.send("Số lượng không hợp lệ.")
            return

    res = await process_smelt(ctx, user_id, ore_key, amount)
    await ctx.send(res)


@bot.command(name='taiche')
async def taiche(ctx, qty: str = '1'):
    """Tái chế: p taiche <số_lượng>  -> mỗi đơn vị dùng 10 xỉ để trả về 1 ore kim loại ngẫu nhiên."""
    user_id = str(ctx.author.id)
    user_data = get_user_data(user_id)
    inv = user_data.get('inventory', {})

    try:
        amount = int(qty) if qty.lower() != 'all' else (inv.get('slag', 0) // 10)
    except Exception:
        await ctx.send("Số lượng không hợp lệ.")
        return

    if amount <= 0:
        await ctx.send("Số lượng phải lớn hơn 0.")
        return

    have_slag = inv.get('slag', 0)
    need = amount * 10
    if have_slag < need:
        await ctx.send(f"Bạn cần {need} xỉ để tái chế {amount} lần. Bạn chỉ có {have_slag} xỉ.")
        return

    # metal ores available to return
    metal_options = ['iron', 'copper', 'gold', 'silver']
    results = {}
    for _ in range(amount):
        chosen = random.choice(metal_options)
        inv[chosen] = inv.get(chosen, 0) + 1
        results[chosen] = results.get(chosen, 0) + 1

    # consume slag
    inv['slag'] = have_slag - need
    if inv['slag'] == 0:
        del inv['slag']

    user_data['inventory'] = inv
    save_data(player_inventory)

    parts = [f"{v} {k}" for k, v in results.items()]
    await ctx.send(f"♻️ Tái chế xong: {ctx.author.mention} đã đổi {need} xỉ thành: {', '.join(parts)}")


# app$.py

@bot.command()
async def sell(ctx, ore_type: str = None, quantity_to_sell: str = None):
    user_id = str(ctx.author.id)
    user_data = get_user_data(user_id)
    inv = user_data['inventory']
    
    if ore_type is None or quantity_to_sell is None:
        await ctx.send(f"{ctx.author.mention} vui lòng nhập <tên quặng> và <số lượng> cần bán (hoặc 'all').")
        return

    ore_type = ore_type.lower()
    base_price = price.get(ore_type)
    
    if base_price is None:
        await ctx.send(f"❌ Không tìm thấy giá cho **{ore_type}**.")
        return
        
    # Lấy tổng số lượng có
    total_available = get_total_ore_count(inv, ore_type)
    
    if quantity_to_sell.lower() == 'all':
        amount_to_sell = total_available
    else:
        try:
            amount_to_sell = int(quantity_to_sell)
        except ValueError:
            await ctx.send("Số lượng không hợp lệ.")
            return

    if amount_to_sell <= 0:
        await ctx.send("Số lượng phải lớn hơn 0.")
        return
        
    if amount_to_sell > total_available:
        await ctx.send(f"Bạn không có đủ **{ore_type}** để bán. Bạn chỉ có **{total_available:,}**.")
        return
        
    # Lấy quặng để bán (Ưu tiên bán quặng chất lượng thấp nhất trước)
    # Hàm remove_ore_units được dùng để lấy ra số lượng cần bán
    removed = remove_ore_units(inv, ore_type, amount_to_sell, strategy='lowest') 
    
    if not removed:
        await ctx.send(f"❌ Bạn không có quặng **{ore_type}** để bán.")
        return

    total_price = 0
    parts = []
    
    # Tính giá dựa trên chất lượng (Áp dụng công thức mới: Giá gốc * Chất lượng%)
    for q_str, cnt in removed.items():
        q = int(q_str) # Chất lượng (từ 15 đến 100)
        
        # === Đã thay đổi từ phép cộng sang phép nhân theo yêu cầu ===
        # Công thức mới: base_price * (q / 100.0)
        price_per_unit = base_price * (q / 100.0)
        
        # Tính tổng tiền và làm tròn xuống số nguyên
        total_price += int(price_per_unit * cnt)
        
        # Cập nhật parts cho thông báo
        parts.append(f"{cnt}x{q}% (Giá: {int(price_per_unit):,} {KESLING_ICON})")
    
    # Cập nhật tiền
    user_data['money'] = user_data.get('money', 0) + total_price
    
    # Lưu dữ liệu
    save_data(player_inventory)
    
    # Gửi thông báo
    await ctx.send(f"{ctx.author.mention} đã bán {sum(removed.values()):,} {ore_type} ({', '.join(parts)}) và nhận được **{total_price:,} {KESLING_ICON}**.")
# --- Hàm Hỗ Trợ Cho Leaderboard Quặng Hiếm ---
def get_rarest_ore_score(inventory, ore_weights):
    """Tính điểm hiếm dựa trên quặng hiếm nhất và số lượng của nó."""
    rarest_score = 0
    rarest_ore_name = None

    player_ores = {
        name: get_total_ore_count(inventory, name)
        for name in inventory.keys()
        if name in ore_weights and name not in ('dirt', 'stone')
    }
    
    for ore_name, count in player_ores.items():
        if count > 0:
            weight = ore_weights[ore_name]
            # Công thức tính điểm: (1000 / Trọng số) * Số lượng quặng
            current_score = (1000 / weight) * count
            
            if current_score > rarest_score:
                rarest_score = current_score
                rarest_ore_name = ore_name

    if rarest_ore_name:
        count = get_total_ore_count(inventory, rarest_ore_name)
        return (rarest_score, rarest_ore_name, count)
    return (0, None, 0)



# --- Hàm Hỗ Trợ Tạo Embed Leaderboard Theo Trang ---
async def create_leaderboard_embed(ctx, bot, ranking_data, lb_type, page_num, total_pages, ore_weights, emoji_icons):
    """Tạo embed cho một trang cụ thể của Leaderboard."""
    
    PER_PAGE = 10
    start_index = (page_num - 1) * PER_PAGE
    end_index = start_index + PER_PAGE
    page_data = ranking_data[start_index:end_index]
    
    ranking_list_on_page = []

    # 1. Khai báo Embed và Tiêu đề
    if lb_type in ("money", "keslings", "tiền"):
        embed = discord.Embed(
            title="🏆 BẢNG XẾP HẠNG TIỀN TỆ 🏆",
            description=f"Top {len(ranking_data)} đại gia giàu nhất server (Trang {page_num}/{total_pages})",
            color=discord.Color.gold()
        )
        # Giữ nguyên field_name_template (nhưng không cần thiết lắm nếu dùng f-string)
        field_name_template = "{medals} {i}. {user_name}"
        # ❌ XÓA DÒNG field_value_template BỊ LỖI Ở ĐÂY
        
        # Tiền tệ có 2 giá trị trong item: (user_id, money)
        for user_id, money in page_data:
            ranking_list_on_page.append((user_id, money))
        
    else: # ore
        embed = discord.Embed(
            title="💎 BẢNG XẾP HẠNG QUẶNG HIẾM 💎",
            description=f"Top {len(ranking_data)} người sở hữu quặng hiếm nhất (Trang {page_num}/{total_pages})",
            color=discord.Color.teal()
        )
        field_name_template = "{medals} {i}. {user_name}"
        # field_value_template = "✨ **{count}x {ore_name}** (Điểm: {score:.0f})" # Giữ nguyên nếu dùng template cho ore
        
        # Quặng có 4 giá trị trong item: (user_id, score, rarest_ore, count)
        for user_id, score, rarest_ore, count in page_data:
            ranking_list_on_page.append((user_id, count, rarest_ore, score))

    # 2. Vòng lặp cuối cùng để fetch user và tạo field cho embed
    for j, item in enumerate(ranking_list_on_page, start=1):
        i = start_index + j # Thứ hạng toàn cục (1, 2, 3, ...)
        user_id = item[0]
        
        try:
            user = await bot.fetch_user(int(user_id))
            user_name = user.name
            
            # Logic huy chương (Giữ nguyên)
            if i == 1:
                medal = "🥇"
            elif i == 2:
                medal = "🥈"
            elif i == 3:
                medal = "🥉"
            elif i <= 10:
                medal = "🏅"
            else:
                medal = "" # Chỉ hiển thị số hạng ở field_name (i)
            
            # Tạo tên Field (chung cho mọi loại LB)
            field_name = f"{medal} {i}. {user_name}"
            
            # 3. Tính toán Giá trị Field (field_value)
            if lb_type in ("money", "keslings", "tiền"):
                # ✅ SỬA LỖI: Sử dụng KESLING_ICON và định dạng f-string trực tiếp
                money = item[1]
                # Sử dụng KESLING_ICON, đã giả định được định nghĩa ở ngoài hàm
                field_value = "💰 **{:,}** {}".format(money, KESLING_ICON)
                
            elif lb_type in ("ore", "quặng"):
                # Logic cho Ore (Sử dụng f-string thay vì template phức tạp)
                ore_name = item[2]
                count = item[1]
                score = item[3]
                
                # SỬA: Định dạng lại ore_name để dùng f-string đơn giản
                ore_display = emoji_icons.get(ore_name, "") + " " + ore_name.upper()
                field_value = f"✨ **{count}x {ore_display}** (Điểm: {score:.0f})"

            # 4. Thêm Field vào Embed
            embed.add_field(
                name=field_name, 
                value=field_value, 
                inline=False
            )
            
        except Exception as e:
            print(f"Lỗi khi fetch user {user_id}: {e}")
            continue

    return embed

    # --- Class View để xử lý nút chuyển trang ---
class LeaderboardView(View):
    def __init__(self, ctx, bot, ranking_data, lb_type, total_pages, ore_weights, emoji_icons, timeout=180):
        super().__init__(timeout=timeout)
        self.ctx = ctx
        self.bot = bot
        self.ranking_data = ranking_data
        self.lb_type = lb_type
        self.total_pages = total_pages
        self.current_page = 1
        self.ore_weights = ore_weights
        self.emoji_icons = emoji_icons
        
        # Khởi tạo trạng thái nút
        self.update_buttons()
        
    def update_buttons(self):
        """Cập nhật trạng thái bật/tắt của các nút."""
        # Nếu chỉ có 1 trang, tắt hết
        if self.total_pages <= 1:
            for child in self.children:
                child.disabled = True
            return

        for child in self.children:
            if child.custom_id == "prev_page":
                child.disabled = (self.current_page == 1)
            elif child.custom_id == "next_page":
                child.disabled = (self.current_page == self.total_pages)

    async def update_embed(self, interaction: discord.Interaction):
        """Tạo embed mới, cập nhật nút và gửi edit message."""
        # Chỉ người dùng gọi lệnh mới được tương tác
        if interaction.user != self.ctx.author:
            return await interaction.response.send_message("❌ Bạn không phải người dùng lệnh này.", ephemeral=True)
            
        new_embed = await create_leaderboard_embed(
            self.ctx, 
            self.bot, 
            self.ranking_data, 
            self.lb_type, 
            self.current_page, 
            self.total_pages,
            self.ore_weights,
            self.emoji_icons
        )
        self.update_buttons()
        await interaction.response.edit_message(embed=new_embed, view=self)

    # Decorator button là cách viết gọn cho việc thêm nút
    @button(label="< Trang Trước", style=ButtonStyle.blurple, custom_id="prev_page")
    async def prev_page(self, interaction: discord.Interaction, button: Button):
        if self.current_page > 1:
            self.current_page -= 1
            await self.update_embed(interaction)

    @button(label="Trang Sau >", style=ButtonStyle.blurple, custom_id="next_page")
    async def next_page(self, interaction: discord.Interaction, button: Button):
        if self.current_page < self.total_pages:
            self.current_page += 1
            await self.update_embed(interaction)

    async def on_timeout(self):
        """Hủy tất cả các nút khi hết thời gian chờ (180s)."""
        # Đây là cách tốt nhất để tắt nút trên message gốc
        for child in self.children:
            child.disabled = True
        try:
            # self.message là message ban đầu, cần được gán trong lệnh lb
            await self.message.edit(view=self)
        except Exception:
            # Bỏ qua nếu tin nhắn bị xóa hoặc không thể chỉnh sửa
            pass
        




# --- Thay thế lệnh lb cũ bằng lệnh mới có phân trang ---
@bot.command()
async def lb(ctx, lb_type: str = "money"):
    """
    Hiển thị Bảng xếp hạng có phân trang.
    Dùng: p lb [money/ore]
    - p lb (mặc định là money)
    - p lb ore (Bảng xếp hạng quặng hiếm)
    """
    global ore # Đảm bảo truy cập được dict ore
    global emoji_icon # Đảm bảo truy cập được dict emoji_icon

    if not player_inventory:
        await ctx.send("📉 Chưa có người chơi nào trong bảng xếp hạng.")
        return

    valid_players = {
        user_id: data
        for user_id, data in player_inventory.items()
        if isinstance(data, dict)
    }

    if not valid_players:
        await ctx.send("📉 Không có dữ liệu người chơi hợp lệ.")
        return

    lb_type = lb_type.lower()
    ranking_data = [] # Lưu trữ toàn bộ dữ liệu xếp hạng

    if lb_type in ("money", "{KESLING_ICON}", "tiền"):
        # --- Bảng xếp hạng Tiền tệ ---
        money_players = {
            user_id: data.get('money', 0)
            for user_id, data in valid_players.items()
        }
        # Lọc người chơi có tiền > 0
        money_ranking = {k: v for k, v in money_players.items() if v > 0}

        if not money_ranking:
            await ctx.send("📉 Không có người chơi nào có dữ liệu tiền tệ.")
            return

        # Sắp xếp theo số tiền giảm dần
        ranking_data = sorted(money_ranking.items(), key=lambda x: x[1], reverse=True)
        
    elif lb_type in ("ore", "quặng"):
        # --- Bảng xếp hạng Quặng Hiếm ---
        for user_id, data in valid_players.items():
            # item: (user_id, score, rarest_ore, count)
            score, rarest_ore, count = get_rarest_ore_score(data.get('inventory', {}), ore)
            if score > 0:
                ranking_data.append((user_id, score, rarest_ore, count))

        if not ranking_data:
            await ctx.send("📉 Không có người chơi nào có quặng hiếm (ngoại trừ dirt/stone).")
            return

        # Sắp xếp theo điểm hiếm giảm dần
        ranking_data = sorted(ranking_data, key=lambda x: x[1], reverse=True)
        
    else:
        await ctx.send("❌ Loại bảng xếp hạng không hợp lệ. Vui lòng dùng `p lb money` hoặc `p lb ore`.")
        return
    
    # --- Logic Phân Trang ---
    PER_PAGE = 10
    total_entries = len(ranking_data)
    total_pages = math.ceil(total_entries / PER_PAGE)
    
    # Tạo embed ban đầu (trang 1)
    initial_embed = await create_leaderboard_embed(
        ctx, bot, ranking_data, lb_type, 1, total_pages, ore, emoji_icon
    )
    
    # Tạo View và gửi tin nhắn
    view = LeaderboardView(
        ctx, bot, ranking_data, lb_type, total_pages, ore, emoji_icon
    )
    
    # Gửi tin nhắn và lưu message để view có thể chỉnh sửa sau timeout
    message = await ctx.send(embed=initial_embed, view=view)
    view.message = message # Gán message object vào view instance



        
# cuối cái ui phân trang
@bot.command(name='listore')
async def listore(ctx):
    """Liệt kê toàn bộ quặng có thể khai thác (dựa trên dict `ore`)."""
    # Đảm bảo bạn có các dict `ore`, `emoji_icon`, `price` đã được định nghĩa
    
    ore_items = []
    # Sắp xếp quặng theo trọng số (weight) để dễ theo dõi
    # Sử dụng `list(ore.items())` để đảm bảo tương thích
    sorted_ore = sorted(ore.items(), key=lambda item: item[1], reverse=True)
    
    for k, weight in sorted_ore:
        emoji = emoji_icon.get(k, '')
        base_price = price.get(k, 'N/A')
        
        # *** FIX LỖI `ValueError: Cannot specify ',' with 's'.` ***
        # Kiểm tra nếu giá là chuỗi 'N/A' hoặc không phải là số
        if isinstance(base_price, (int, float)):
            # Nếu là số, định dạng có dấu phẩy
            price_display = f"{base_price:,} {KESLING_ICON}s"
        elif isinstance(base_price, str) and base_price.isdigit():
            # Nếu là chuỗi số, chuyển thành int và định dạng
            price_display = f"{int(base_price):,} {KESLING_ICON}s"
        else:
            # Nếu là 'N/A' hoặc chuỗi khác, không định dạng
            price_display = f"{base_price} {KESLING_ICON}s"
            
        # Định dạng dòng: Emoji TênQuặng (Trọng số W | Giá: P {KESLING_ICON}s)
        ore_items.append(f"{emoji} **{k}** (Trọng số: {weight} | Giá: {price_display})")

    # Sửa lỗi 400: Sử dụng hàm phân trang (đã được sửa)
    await send_paginated_via_ctx(ctx, "Danh Sách Quặng Có Thể Khai Thác", ore_items, per_page=15)



@bot.command()
@commands.has_permissions(administrator=True)
async def say(ctx, channel: discord.TextChannel = None, *, message: str = None):
    if channel is None or message is None:
        await ctx.send(
            f"{ctx.author.mention} cách dùng lệnh:\n`!say #tên-kênh nội dung`\nVí dụ: `!say #general Hello mọi người!`"
        )
        return

    await channel.send(message)
    await ctx.send(f"✅ Đã gửi tin nhắn đến {channel.mention}")
@say.error
async def say_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send(f"{ctx.author.mention} bạn không có quyền dùng lệnh này.")
         
            
@bot.command(name="mod")
async def mod(ctx, action: str = None, member: discord.Member = None, target: str = None, amount: int = None):
    if str(ctx.author.id) != owner_id and str(ctx.author.id) not in subowner_id:
        await ctx.send("🚫 Bạn không có quyền dùng lệnh này.")
        return

    # Hiển thị hướng dẫn nếu thiếu tham số
    if not action or not member or not target or amount is None:
        embed = discord.Embed(
            title="📘 Hướng dẫn sử dụng lệnh pmod",
            description="Lệnh quản trị để thêm, xóa, hoặc đặt lại tiền/quặng cho người chơi.",
            color=discord.Color.blue()
        )
        embed.add_field(name="Thêm tiền", value="`p pmod addmoney @user 0 <số_tiền>`", inline=False)
        embed.add_field(name="Đặt lại tiền", value="`p pmod setmoney @user 0 <số_tiền_mới>`", inline=False)
        embed.add_field(name="Thêm quặng", value="`p pmod addore @user <tên_quặng> <số_lượng>`", inline=False)
        embed.add_field(name="Xóa quặng", value="`p pmod removeore @user <tên_quặng> <số_lượng>`", inline=False)
        embed.set_footer(text="Chỉ người có quyền OP mới dùng được lệnh này.")
        await ctx.send(embed=embed)
        return

    user_id = str(member.id)
    user_data = get_user_data(user_id)
    action = action.lower()
    target = target.lower()

    if action == "addmoney":
        user_data['money'] = user_data.get('money', 0) + amount
        save_data(player_inventory)  # Chỉ lưu lại toàn bộ player_inventory
        await ctx.send(f"✅ Đã thêm {amount} {KESLING_ICON}s cho {member.display_name}. Tổng: {user_data['money']} {KESLING_ICON}s.")

    elif action == "setmoney":
        user_data['money'] = amount
        save_data(player_inventory)  # Chỉ lưu lại toàn bộ player_inventory
        await ctx.send(f"✅ Đã đặt lại tiền của {member.display_name} thành {amount} {KESLING_ICON}s.")

    elif action == "addore":
        if target not in ore:
            await ctx.send("❌ Quặng không tồn tại.")
            return
        inv = user_data.get('inventory', {})
        user_data['inventory'] = inv
        inv[target] = inv.get(target, 0) + amount
        save_data(player_inventory)  # Chỉ lưu lại toàn bộ player_inventory
        emoji = emoji_icon.get(target, "")
        await ctx.send(f"✅ Đã thêm {amount} {emoji} {target} cho {member.display_name}.")

    elif action == "removeore":
        if target not in ore:
            await ctx.send("❌ Quặng không tồn tại.")
            return
        inv = user_data.get('inventory', {})
        current = inv.get(target, 0)
        if current < amount:
            await ctx.send(f"❌ {member.display_name} chỉ có {current} {target}, không thể xóa {amount}.")
            return
        inv[target] = current - amount
        save_data(player_inventory)  # Chỉ lưu lại toàn bộ player_inventory
        emoji = emoji_icon.get(target, "")
        await ctx.send(f"✅ Đã xóa {amount} {emoji} {target} khỏi kho của {member.display_name}.")

    else:
        await ctx.send("❌ Hành động không hợp lệ. Dùng `addmoney`, `setmoney`, `addore`, hoặc `removeore`.")
        
          
        
        
        
        
# Lệnh daily
@bot.command()
async def daily(ctx):
    user_id = str(ctx.author.id)
    # Use the in-memory player_inventory so other commands see updates immediately
    user_data = get_user_data(user_id)

    # Use UTC naive datetime consistently
    times = datetime.datetime.utcnow()
    last_claim = user_data.get("last_claim")

    if last_claim:
        try:
            last_time = datetime.datetime.fromisoformat(last_claim)
            if (times - last_time).days < 1:
                await ctx.send("❌ Hôm nay bạn đã nhận rồi, mai quay lại nhé!")
                return
        except Exception:
            # If parsing fails, allow claim and overwrite last_claim
            pass

    # Cộng tiền
    amount = random.randint(20, 200)
    user_data["money"] = user_data.get("money", 0) + amount
    user_data["last_claim"] = times.isoformat()
    await ctx.send(f"💰 bạn nhận đc {amount} {KESLING_ICON}s , mai nhớ check đấy!")

    # Xác suất nhận quặng
    if random.random() < 0.1:
        ore_name = random.choice(list(ore.keys()))
        inv = user_data["inventory"]
        inv[ore_name] = inv.get(ore_name, 0) + 1
        emoji = emoji_icon.get(ore_name, "")
        await ctx.send(f"🎉 May mắn dữ ta! Bạn nhận được 1 {emoji} {ore_name} quặng.")

    # Xác suất nhận ticket
    if random.random() < 0.02:
        inv = user_data["inventory"]
        inv["ticket"] = inv.get("ticket", 0) + 1
        await ctx.send(f"🎫 nhân phẩm cao đấy! {ctx.author.mention} Nhận được 1 ticket.")

    # Lưu lại dữ liệu
    # Lưu lại vào player_inventory và file để các lệnh khác đọc được ngay
    player_inventory[user_id] = user_data
    save_data(player_inventory)

    
    
    
    
    
@bot.command()
async def tktocoin(ctx):
    user_id = str(ctx.author.id)
    data = load_data()
    user_data = data.get(user_id, {"money": 0, "inventory": {}})
    inv = user_data["inventory"]

    if inv.get("ticket", 0) < 1:
        await ctx.send("❌ Bạn không có ticket nào để đổi.")
        return

    # Trừ ticket và cộng tiền ngẫu nhiên
    inv["ticket"] -= 1
    amount = random.randint(200, 5000)
    user_data["money"] += amount
    await ctx.send(f"💸 Bạn đã đổi 1 ticket lấy {amount} {KESLING_ICON}s!")

    data[user_id] = user_data
    save_data(data)

@bot.command()
async def hlp(ctx):
    embed = discord.Embed(
        title="📜 Danh sách lệnh bot",
        description="Dưới đây là các lệnh bạn có thể dùng:",
        color=discord.Color.blue()
    )

    embed.add_field(name="`pdaily`", value="Nhận tiền và phần thưởng mỗi ngày ,có 10% nhận đc quặng và 2% ra ticket", inline=False)
    embed.add_field(name="`ptktocoin`", value="Đổi 1 ticket thành {KESLING_ICON} ngẫu nhiên (200–5000)", inline=False)
    embed.add_field(name="`pbag`", value="Xem kho vật phẩm/quặng", inline=False)
    embed.add_field(name="pmine", value="đào quặng", inline=False)
    embed.add_field(name="psell", value="psell <tên quặng> <số lượng/all>", inline=False)
    embed.add_field(name="pcf", value="pcf <h/t> < số tiền cược >", inline=False)
    embed.add_field(name="plb", value="xem leaderboard thôi", inline=False)
    embed.add_field(name="phlp", value="thứ bạn đang xem hiện tại đấy", inline=False)

    embed.set_footer(text="Bot by phc bot#2136 • Dùng lệnh mỗi ngày để nhận quà 🎁")

    await ctx.send(embed=embed)
@bot.command(name="give", help="Chuyển tiền cho người chơi khác.")
async def give_money(ctx, member: discord.Member, amount: int):
    """
    Chuyển một lượng tiền (amount) từ người gửi (ctx.author) 
    sang người nhận (member).
    """
    
    # --- 1. Kiểm tra điều kiện ---
    if amount <= 0:
        return await ctx.send("Số tiền cần chuyển phải lớn hơn 0.")
        
    user_id = str(ctx.author.id)
    recipient_id = str(member.id)

    # Không thể tự chuyển cho chính mình
    if user_id == recipient_id:
        return await ctx.send("Bạn không thể tự chuyển tiền cho chính mình.")

    # --- 2. Tải dữ liệu ---
    # *Sử dụng hàm load_data/get_user_data của bạn*
    player_inventory = load_data() # (Sử dụng tạm hàm JSON cũ, hoặc hàm SQL mới ở mục 2)
    
    sender = player_inventory.get(user_id, {})
    sender_money = sender.get("money", 0)

    # --- 3. Kiểm tra số dư người gửi ---
    if sender_money < amount:
        return await ctx.send(f"❌ Bạn không đủ **{amount:,} {KESLING_ICON}s** để chuyển. Số dư hiện tại của bạn là **{sender_money:,} {KESLING_ICON}s**.")

    # --- 4. Thực hiện chuyển khoản ---
    recipient = player_inventory.get(recipient_id, {})
    recipient_money = recipient.get("money", 0)
    
    # Cập nhật số tiền
    sender["money"] = sender_money - amount
    recipient["money"] = recipient_money + amount
    
    player_inventory[user_id] = sender
    player_inventory[recipient_id] = recipient
    
    # *Sử dụng hàm save_data/update_user_money của bạn*
    save_data(player_inventory) # (Sử dụng tạm hàm JSON cũ, hoặc hàm SQL mới ở mục 2)

    # --- 5. Gửi thông báo ---
    await ctx.send(f"✅ **{ctx.author.name}** đã chuyển thành công **{amount:,} {KESLING_ICON}s** cho **{member.display_name}**.")
class MoneyModal(discord.ui.Modal, title="Đặt cược"):
    amount = discord.ui.TextInput(
        label="Nhập số tiền bạn muốn cược",
        placeholder="Ví dụ: 1000",
        required=True,
        max_length=10
    )
    
    def __init__(self, view):
        super().__init__()
        self.parent_view = view
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            bet = int(self.amount.value.strip())
        except Exception:
            await interaction.response.send_message("Số tiền không hợp lệ.", ephemeral=True)
            return
        
        player_inventory = load_data()
        user_id = str(interaction.user.id)
        player = player_inventory.get(user_id, {})
        current_money = player.get("money", 0)
        
        if bet <= 0:
            await interaction.response.send_message("Cược phải lớn hơn 0.", ephemeral=True)
            return
        
        if bet > current_money:
            await interaction.response.send_message(f"Bạn không đủ {self.parent_view.money_icon}s. Bạn chỉ có {current_money:,}.", ephemeral=True)
            return
            
        # Trừ tiền cược
        player["money"] = current_money - bet
        player_inventory[user_id] = player
        save_data(player_inventory)
        
        self.parent_view.bet = bet
        
        await interaction.response.defer()
        await self.parent_view.start_game(interaction)


class NumberModal(discord.ui.Modal, title="Cược số chính xác (3-18)"):
    amount = discord.ui.TextInput(
        label="Số cược (3-18)",
        placeholder="Ví dụ: 8",
        required=True,
        max_length=2
    )
    
    def __init__(self, view):
        super().__init__()
        self.parent_view = view
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            number = int(self.amount.value.strip())
        except Exception:
            await interaction.response.send_message("Số cược không hợp lệ.", ephemeral=True)
            return

        # Sửa giới hạn số cược từ 3 đến 18
        if not (3 <= number <= 18):
            await interaction.response.send_message("Chỉ cược từ 3 đến 18!", ephemeral=True)
            return

        self.parent_view.choice = str(number)
        await interaction.response.defer()
        await self.parent_view.end_choice(interaction) # Chuyển sang nhập tiền cược


class TaiXiuView(discord.ui.View):
    def __init__(self, author, money_icon):
        super().__init__(timeout=90)
        self.author = author
        self.money_icon = money_icon
        self.bet = None
        self.choice = None  # Khởi tạo None để fix lỗi AttributeError
        self.message = None
        
    async def on_timeout(self):
        # Tắt tất cả các nút khi hết thời gian
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(content="Đã hết thời gian cược Tài Xỉu.", view=self)

    async def end_choice(self, interaction: discord.Interaction):
        if interaction.user != self.author:
            await interaction.response.send_message("❌ Không phải lượt của bạn!", ephemeral=True)
            return

        # Tắt các nút chọn
        for item in self.children:
            item.disabled = True
            
        await interaction.message.edit(view=self)
        
        # Mở MoneyModal để nhập tiền cược
        modal = MoneyModal(self)
        await interaction.response.send_modal(modal)

    # --- BUTTONS (Hàng 1) ---
    @discord.ui.button(label="TÀI (11-17)", style=discord.ButtonStyle.secondary, emoji="⬆️", row=0)
    async def tai(self, interaction: discord.Interaction, button):
        self.choice = "tai"
        await self.end_choice(interaction)

    @discord.ui.button(label="XỈU (4-10)", style=discord.ButtonStyle.secondary, emoji="⬇️", row=0)
    async def xiu(self, interaction: discord.Interaction, button):
        self.choice = "xiu"
        await self.end_choice(interaction)

    @discord.ui.button(label="CHẴN", style=discord.ButtonStyle.secondary, emoji="⚫", row=0)
    async def chan(self, interaction: discord.Interaction, button):
        self.choice = "chan"
        await self.end_choice(interaction)

    @discord.ui.button(label="LẺ", style=discord.ButtonStyle.secondary, emoji="🔴", row=0)
    async def le(self, interaction: discord.Interaction, button):
        self.choice = "le"
        await self.end_choice(interaction)

    # --- BUTTONS (Hàng 2) ---
    @discord.ui.button(label="CƯỢC SỐ (3-18)", style=discord.ButtonStyle.primary, emoji="🎯", row=1)
    async def number(self, interaction: discord.Interaction, button):
        modal = NumberModal(self)
        await interaction.response.send_modal(modal)


    async def start_game(self, interaction: discord.Interaction):
        # Chuẩn bị embed
        embed = discord.Embed(title="🎲 Tài Xỉu", color=discord.Color.gold())
        
        # Sửa lỗi: kiểm tra None trước khi gọi .upper()
        choice_display = self.choice.upper() if self.choice else "Lựa chọn bị lỗi"
        
        embed.add_field(name="Cược", value=f"**{choice_display}** ({self.bet:,} {self.money_icon}s)", inline=False)
        embed.add_field(name="Xúc xắc", value="`? + ? + ?`", inline=False)
        embed.set_footer(text=f"Người chơi: {self.author.name}")

        msg = await interaction.edit_original_response(embed=embed, view=None)

        # Animation
        dice = [0, 0, 0]
        emojis = ['⚀', '⚁', '⚂', '⚃', '⚄', '⚅']
        
        for i in range(3):
            await asyncio.sleep(1.2)
            dice[i] = random.randint(1, 6)
            current = " + ".join(emojis[d-1] if d > 0 else "?" for d in dice)
            embed.set_field_at(1, name="Xúc xắc", value=f"`{current}`", inline=False)
            await msg.edit(embed=embed)
            
        total = sum(dice)
        is_triple = len(set(dice)) == 1 # Kiểm tra Bộ Ba
        
        # Xử lý thắng thua
        win = False
        multiplier = 2 # Mặc định x2 cho Tài/Xỉu/Chẵn/Lẻ
        
        # 1. Luật Bộ Ba (Triples)
        if is_triple:
            if self.choice in ['tai', 'xiu', 'chan', 'le']:
                win = False # Thua khi ra Bộ Ba
            elif self.choice == str(total): 
                win = True
                multiplier = 10 # Cược số chính xác 3 hoặc 18 vẫn thắng x10
        
        # 2. Tài hoặc Xỉu (chỉ xét nếu không phải Bộ Ba)
        elif self.choice in ['tai', 'xiu']:
            # Tài (11-17), Xỉu (4-10)
            win = (self.choice == 'tai' and total >= 11) or (self.choice == 'xiu' and total <= 10)
            
        # 3. Chẵn hoặc Lẻ (chỉ xét nếu không phải Bộ Ba)
        elif self.choice in ['chan', 'le']:
            win = (self.choice == 'chan' and total % 2 == 0) or (self.choice == 'le' and total % 2 == 1)
            
        # 4. Cược Số Chính Xác (3-18)
        else:
            win = total == int(self.choice)
            multiplier = 10 
            
        # Tính tiền thưởng và cập nhật tài khoản
        prize = self.bet * multiplier if win else 0
        player_inventory = load_data()
        user_id = str(self.author.id)
        player = player_inventory.get(user_id, {})
        
        # Tiền đã trừ trước đó, giờ cộng lại tiền thắng (prize)
        player['money'] += prize
        save_data(player_inventory) 
        
        # Kết quả hiển thị
        result_text = f"{'✨ THẮNG' if win else '👎 THUA'}! Tổng: **{total}** (`{'TÀI' if total >= 11 else 'XỈU'}`, `{'CHẴN' if total % 2 == 0 else 'LẺ'}`)."
        
        if is_triple:
            result_text += f"\n🎲 **ĐÃ RA BỘ BA** ({dice[0]}-{dice[0]}-{dice[0]})."
            
        if win:
            result_text += f"\nBạn thắng **{prize:,} {self.money_icon}s**!"
        else:
            result_text += f"\nBạn mất **{self.bet:,} {self.money_icon}s**."

        embed.set_field_at(1, name="KẾT QUẢ", value=f"{result_text}\n\n**Tiền hiện có:** {player['money']:,} {self.money_icon}s", inline=False)
        embed.color = discord.Color.green() if win else discord.Color.red()
        await msg.edit(embed=embed)

# === LỆNH CHÍNH ===
# File: app$.py

@bot.command(name='taixiu', aliases=['tx'])
async def taixiu_command(ctx, amount: Optional[int] = None):
    user_id = str(ctx.author.id)
    player_inventory = load_data()
    player = player_inventory.get(user_id, {})
    current_money = player.get("money", 0)

    # Initial embedded message
    embed = discord.Embed(
        title="🎲 Tài Xỉu",
        description="Chọn cửa bạn muốn cược:\n"
                    "**TÀI** (11-17), **XỈU** (4-10), **CHẴN**, **LẺ** (Tất cả đều thua khi ra Bộ Ba)\n"
                    "**CƯỢC SỐ** (3-18) thắng x10.",
        color=discord.Color.blue()
    )
    embed.add_field(name="Tiền hiện có", value=f"{current_money:,} {KESLING_ICON}s", inline=False)
    embed.set_footer(text=f"Người chơi: {ctx.author.name} | Bạn có 90 giây để chọn cửa cược.")
    
    view = TaiXiuView(ctx.author, KESLING_ICON)
    view.message = await ctx.send(embed=embed, view=view)



@bot.command()
async def quang(ctx, *, ten_quang):
    # Tạo URL tìm kiếm hình ảnh trên Google hoặc Bing
    url = f"https://www.bing.com/search?q={ten_quang}"

    # Gửi link hình ảnh cho người dùng
    await ctx.send(f"🔍 Đây là hình ảnh về **{ten_quang}**: {url}")


@bot.command()
async def checkgia(ctx, ore_name: str = None):
    """Kiểm tra giá của một loại quặng. Dùng: pcheckgia <tên_quặng>"""
    if ore_name is None:
        await ctx.send(f"{ctx.author.mention} Vui lòng nhập tên quặng. Cách dùng: `pcheckgia <tên_quặng>`")
        return

    key = ore_name.lower()
    if key not in price:
        await ctx.send(f"{ctx.author.mention} Không tìm thấy quặng '{ore_name}'. Hãy thử tên khác.")
        return

    emoji = emoji_icon.get(key, "")
    unit_price = price[key]
    await ctx.send(f"✅ Giá của {emoji} **{key}** là **{unit_price} {KESLING_ICON}s / 1 unit.")

async def send_paginated_via_ctx(ctx, title: str, lines: list[str], per_page: int = 10):
    """Gửi danh sách dài bằng nhiều embed để tránh lỗi/gói tin quá lớn."""
    if not lines:
        await ctx.send("Không có dữ liệu để hiển thị.")
        return
    pages = [lines[i:i+per_page] for i in range(0, len(lines), per_page)]
    for i, page in enumerate(pages, start=1):
        embed = discord.Embed(title=f"{title} (Trang {i}/{len(pages)})", color=discord.Color.blue())
        embed.description = "\n".join(page)
        await ctx.send(embed=embed)


async def send_paginated_via_interaction(interaction: discord.Interaction, title: str, lines: list[str], per_page: int = 10):
    """Gửi nhiều followup cho slash command (đã defer trước đó)."""
    if not lines:
        await interaction.followup.send("Không có dữ liệu để hiển thị.")
        return
    pages = [lines[i:i+per_page] for i in range(0, len(lines), per_page)]
    for i, page in enumerate(pages, start=1):
        embed = discord.Embed(title=f"{title} (Trang {i}/{len(pages)})", color=discord.Color.blue())
        embed.description = "\n".join(page)
        await interaction.followup.send(embed=embed)


@bot.command(name='help')
async def dynamic_help(ctx, cmd_name: str = None):
    """Tự động liệt kê các lệnh prefix và app (slash). Dùng p help <command> để xem chi tiết."""
    if cmd_name:
        # search prefix commands
        c = discord.utils.get(bot.commands, name=cmd_name)
        if c:
            embed = discord.Embed(title=f"Chi tiết lệnh p{c.name}", description=c.help or "Không có mô tả.")
            embed.add_field(name="Usage", value=f"p{c.name} {c.params if hasattr(c,'params') else ''}")
            await ctx.send(embed=embed)
            return
        # search app commands
        for app in bot.tree.get_commands():
            if app.name == cmd_name:
                desc = app.description or "Không có mô tả."
                await ctx.send(f"/{app.name} - {desc}")
                return
        await ctx.send("Không tìm thấy lệnh.")
        return

    # list prefix commands as lines
    prefix_cmds = [c for c in bot.commands]
    prefix_lines = [f"p{c.name} - {c.help or 'No description'}" for c in prefix_cmds]

    # list slash/app commands as lines
    app_cmds = bot.tree.get_commands()
    app_lines = [f"/{a.name} - {a.description or 'No description'}" for a in app_cmds]

    # Send paginated to prevent oversized single message/embed
    if prefix_lines:
        await send_paginated_via_ctx(ctx, "Prefix commands", prefix_lines, per_page=12)
    else:
        await ctx.send("Không có lệnh prefix.")

    if app_lines:
        await send_paginated_via_ctx(ctx, "App (slash) commands", app_lines, per_page=12)
    else:
        await ctx.send("Không có lệnh app/slash.")


@bot.tree.command(name='help', description='Hiển thị danh sách lệnh và mô tả (prefix + slash)')
async def app_help(interaction: discord.Interaction, command: Optional[str] = None):
    await interaction.response.defer()
    if command:
        c = discord.utils.get(bot.commands, name=command)
        if c:
            await interaction.followup.send(f"p{c.name} - {c.help or 'No description'}")
            return
        for app in bot.tree.get_commands():
            if app.name == command:
                await interaction.followup.send(f"/{app.name} - {app.description or 'No description'}")
                return
        await interaction.followup.send("Không tìm thấy lệnh.")
        return

    prefix_cmds = [c for c in bot.commands]
    prefix_lines = [f"p{c.name} - {c.help or 'No description'}" for c in prefix_cmds]
    app_cmds = bot.tree.get_commands()
    app_lines = [f"/{a.name} - {a.description or 'No description'}" for a in app_cmds]

    if prefix_lines:
        await send_paginated_via_interaction(interaction, "Prefix commands", prefix_lines, per_page=12)
    else:
        await interaction.followup.send("Không có lệnh prefix.")

    if app_lines:
        await send_paginated_via_interaction(interaction, "App (slash) commands", app_lines, per_page=12)
    else:
        await interaction.followup.send("Không có lệnh app/slash.")







# LỆNH TRIEUPHU ĐÃ CẬP NHẬT (THAY THẾ TOÀN BỘ LỆNH CŨ)

class QuizView(discord.ui.View):
    def __init__(self, correct_index, timeout=40.0):
        super().__init__(timeout=timeout)
        self.correct_index = correct_index
        self.choice = None
        self.message = None

        # Tên nút (tùy chọn A, B, C, D)
        labels = ['A', 'B', 'C', 'D']
        
        for i in range(4):
            button = discord.ui.Button(
                label=labels[i], 
                style=discord.ButtonStyle.secondary, 
                custom_id=str(i)
            )
            button.callback = self.create_callback(i)
            self.add_item(button)

    def create_callback(self, index):
        async def callback(interaction: discord.Interaction):
            if interaction.user != self.message.author:
                await interaction.response.send_message("❌ Đây không phải trò chơi của bạn!", ephemeral=True)
                return
            
            self.choice = index
            self.stop()
            await interaction.response.defer() # Phản hồi ngay lập tức
            
        return callback

    async def wait_for_choice(self):
        await self.wait()
        # Vô hiệu hóa tất cả các nút khi có người chọn hoặc hết giờ
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)
        return self.choice


@bot.command(name="trieuphu")
@commands.cooldown(rate=1, per=60, type=commands.BucketType.user) # 10 phút / lần chơi
async def trieuphu(ctx):
    user_id = str(ctx.author.id)
    
    # 🚨 ĐỊA CHỈ API CẦN CẬP NHẬT 🚨
    # Trong lệnh trieuphu của app$.py
    API_URL = "http://paloma.hidencloud.com:12000/api/quiz"

    won_amount = 0
    current_level = 0
    total_levels = len(PRIZE_TIERS) # Mặc định là 10 câu

    # --- 1. LẤY CÂU HỎI TỪ API ---
    async with aiohttp.ClientSession() as session:
        try:
            
            async with session.get(API_URL + f"?amount={total_levels}") as resp:
                if resp.status != 200:
                    await ctx.send(f"❌ Lỗi kết nối API quiz: HTTP {resp.status}")
                    return
                
                questions_data = await resp.json()
                
                # Kiểm tra nếu API trả về lỗi
                if isinstance(questions_data, dict) and "error" in questions_data:
                     await ctx.send(f"❌ Lỗi API: {questions_data['error']}")
                     return

                if not questions_data or len(questions_data) < total_levels:
                    await ctx.send(f"❌ API quiz không cung cấp đủ {total_levels} câu hỏi.")
                    return
                
                
        except Exception as e:
            await ctx.send(f"❌ Lỗi trong quá trình kết nối đến API: {e}")
            return

    await ctx.send(f"💰 **{ctx.author.name}** bắt đầu trò chơi Triệu Phú (10 câu hỏi)! Trả lời sai sẽ kết thúc game.")
    await asyncio.sleep(1.0)
    
    for q in questions_data:
        if current_level >= total_levels:
            break

        level_prize = PRIZE_TIERS[current_level]
        
        # ⚠️ GIẢI MÃ HTML ENTITIES CHO CÂU HỎI VÀ ĐÁP ÁN ⚠️
        question_text = html.unescape(q["question"])
        choices_text = [html.unescape(c) for c in q["choices"]]
        
        correct_index = q["correct_index"]
        
        # Tạo Embed cho câu hỏi
        q_embed = discord.Embed(
            title=f"Câu {current_level + 1} ({level_prize:,} {KESLING_ICON}s) 🏆",
            description=f"**{question_text}**\n\n",
            color=discord.Color.blue()
        )
        
        choices_str = ""
        for i, choice in enumerate(choices_text):
            choices_str += f"**{chr(65+i)}.** {choice}\n" # A, B, C, D
            
        q_embed.add_field(name="Lựa Chọn:", value=choices_str, inline=False)
        q_embed.set_footer(text=f"Mốc an toàn hiện tại: {won_amount} {KESLING_ICON}s | Hết giờ sau 40 giây.")

        # Gửi câu hỏi và chờ câu trả lời
        view = QuizView(correct_index=correct_index)
        message = await ctx.send(embed=q_embed, view=view)
        view.message = message

        idx = await view.wait_for_choice()
        
        if idx is None:
            await ctx.send(f"⏰ Hết giờ — Trò chơi kết thúc. Bạn nhận được: **{won_amount:,} {KESLING_ICON}s**.")
            return

        if idx == correct_index:
            won_amount = PRIZE_TIERS[current_level]
            current_level += 1
            await ctx.send(f"✅ Chính xác! Bạn đã đạt mốc **{won_amount:,} {KESLING_ICON}s**.")
            await asyncio.sleep(1.0)
            continue
        else:
            correct_text = choices_text[correct_index]
            await ctx.send(f"❌ Sai rồi! Đáp án đúng: **{correct_text}**. Bạn rời khỏi cuộc chơi với **{won_amount:,} {KESLING_ICON}s**.")
            break

    # --- 3. KẾT THÚC TRÒ CHƠI VÀ TRAO THƯỞNG ---
    if current_level > 0:
        # Lấy số tiền thắng cuối cùng (won_amount đã được cập nhật hoặc mốc an toàn cuối)
        final_prize = won_amount
        
        # Trao thưởng
        player_inventory = load_data()
        player = player_inventory.get(user_id, {})
        player["money"] = player.get("money", 0) + final_prize
        player_inventory[user_id] = player
        save_data(player_inventory)

        if current_level >= total_levels:
            await ctx.send(f"🏆 Chúc mừng! Bạn đã trả lời đúng cả {total_levels} câu và nhận tổng cộng **{final_prize:,} {KESLING_ICON}s**!")
        else:
            # Trao thưởng khi dừng chơi sớm hoặc trả lời sai
            await ctx.send(f"🎉 Trò chơi kết thúc! **{ctx.author.name}** đã thắng **{final_prize:,} {KESLING_ICON}s** và số tiền này đã được thêm vào tài khoản của bạn.")
    else:
        # Trường hợp sai ngay câu đầu tiên (won_amount vẫn là 0)
        await ctx.send("😭 Rất tiếc, bạn đã trả lời sai ngay câu đầu tiên. Bạn không nhận được tiền thưởng nào.")

        
def run_app_bot():
    
    initialize_vector_db()
    bot.run(TOKEN)

if __name__ == '__main__':
    run_app_bot()