import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from collections import Counter
from ultralytics import YOLO


st.set_page_config(
    page_title="Mister Donut Detector",
    page_icon="🍩",
    layout="wide"
)


# =====================================================
# ตั้งค่า
# =====================================================

MODEL_PATH = "best.pt"
IMGSZ = 800
IOMA_THRESHOLD = 0.60
DISPLAY_MAX = 420


# =====================================================
# ราคาสินค้า
# =====================================================

PRICE_MAP = {
    "Honey Dip": 29,
    "Chocolate ring": 39,
    "Strawberry ring": 32,
    "Choccoconut cake": 29,
    "Coconut cake": 29,
    "Choc peanut cake": 29,
    "Peanut cake": 29,
    "Twist GlaZe": 29,
    "Trio": 29,
    "Blueberry shell": 29,
    "Strawberry shell": 32,
    "Bavarian shell": 29,
    "White Choc PDR": 39,
    "Choc PDR": 39,
    "Glaze PDR": 29,
    "Strawberry PDR": 32,
    "Blueberry PDR": 29,
    "Plain PDR": 29,
}


# =====================================================
# โปรโมชั่น
# =====================================================

PROMOTION_TIERS = [
    (5, 1),
    (7, 2),
    (9, 4),
    (10, 5),
]


def get_promotion(total):
    best = None
    for pay, free in PROMOTION_TIERS:
        required = pay + free
        if total >= required:
            if best is None or required > best[0] + best[1]:
                best = (pay, free)
    return best


def get_next_promotion(total):
    for pay, free in PROMOTION_TIERS:
        required = pay + free
        if total < required:
            return {
                "pay": pay,
                "free": free,
                "required": required,
                "extra": required - total,
            }
    return None


def calculate_free_items(items, free_count):
    sorted_items = sorted(items, key=lambda x: x[0])
    free_items = sorted_items[:free_count]
    return Counter(name for price, name in free_items)


# =====================================================
# CSS
# =====================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.2rem;
    }

    h3 {
        font-size: 1.15rem !important;
        margin-top: 0.2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.10);
        border-radius: 12px;
        padding: 10px 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 10px;
    }

    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# โหลดโมเดล
# =====================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# =====================================================
# ฟังก์ชันตัดกรอบซ้อน
# =====================================================

def remove_overlaps(r, thr=IOMA_THRESHOLD):
    if len(r.boxes) == 0:
        return r

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()

    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = []

    for i in conf.argsort()[::-1]:
        ok = True
        for j in keep:
            ix1 = max(xyxy[i, 0], xyxy[j, 0])
            iy1 = max(xyxy[i, 1], xyxy[j, 1])
            ix2 = min(xyxy[i, 2], xyxy[j, 2])
            iy2 = min(xyxy[i, 3], xyxy[j, 3])

            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            smaller = max(min(area[i], area[j]), 1e-6)

            if inter / smaller > thr:
                ok = False
                break

        if ok:
            keep.append(i)

    return r[sorted(keep)]


# =====================================================
# ย่อรูปสำหรับแสดงผล
# =====================================================

def to_display(arr_or_img, max_side=DISPLAY_MAX):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    im = im.copy()
    im.thumbnail((max_side, max_side))
    return im


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.header("⚙️ ตั้งค่า")

    conf = st.slider(
        "Confidence",
        0.10,
        0.90,
        0.40,
        0.05,
        help="ยิ่งสูง ยิ่งเข้มงวด"
    )

    with st.expander("ขั้นสูง"):
        iou = st.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05)
        ioma = st.slider("IoMA (ตัดกรอบซ้อนใน)", 0.30, 0.95, IOMA_THRESHOLD, 0.05)

    with st.expander("💰 ราคา / โปรโมชั่น"):
        st.caption("ราคาต่อชิ้น")
        for name, price in PRICE_MAP.items():
            st.write(f"• {name}: {price} บาท")

        st.divider()

        for pay, free in PROMOTION_TIERS:
            st.write(f"• ซื้อ {pay} แถม {free} ({pay + free} ชิ้น)")


# =====================================================
# Header + Input
# =====================================================

st.title("🍩 Mister Donut Detector")

st.caption(
    "อัปโหลดหรือถ่ายรูปโดนัท "
    "ระบบจะตรวจจับ นับจำนวน และคำนวณโปรโมชั่นให้"
)


with st.container(border=True):
    tab_upload, tab_camera = st.tabs(["📁 อัปโหลดรูป", "📷 ถ่ายรูป"])

    with tab_upload:
        up_file = st.file_uploader(
            "เลือกรูปโดนัท",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    with tab_camera:
        cam_file = st.camera_input(
            "ถ่ายรูปโดนัท",
            label_visibility="collapsed"
        )


file = up_file or cam_file


if not file:
    st.info("เริ่มต้นโดยอัปโหลดรูปหรือถ่ายรูปโดนัทด้านบน")
    st.stop()


# =====================================================
# ตรวจจับภาพ
# =====================================================

img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
img_bgr = np.array(img)[:, :, ::-1]


with st.spinner("กำลังตรวจจับ..."):
    r = model.predict(
        img_bgr,
        imgsz=IMGSZ,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        max_det=100,
        verbose=False,
    )[0]

    r = remove_overlaps(r, ioma)


plotted_rgb = r.plot()[:, :, ::-1]

counts = Counter(r.names[int(b.cls[0])] for b in r.boxes)

unknown = [name for name in counts if name not in PRICE_MAP]

if unknown:
    st.warning(f"คลาสที่ยังไม่มีราคาใน PRICE_MAP: {unknown}")

file_id = getattr(file, "file_id", None) or f"{file.name}_{file.size}"


# =====================================================
# Layout หลัก
# =====================================================

left, right = st.columns([3, 2], gap="large")


# =====================================================
# ฝั่งซ้าย
# =====================================================

with left:
    st.subheader("ผลการตรวจจับ")
    c1, c2 = st.columns(2)

    with c1:
        st.image(to_display(img), caption="ต้นฉบับ")

    with c2:
        st.image(to_display(plotted_rgb), caption=f"ตรวจพบ {len(r.boxes)} ชิ้น")

    with st.expander("✏️ ตรวจสอบ/แก้จำนวนแต่ละรส", expanded=True):
        st.caption(
            "โมเดลอาจนับผิดรสหรือขาด/เกิน "
            "แก้จำนวนตรงนี้ได้ก่อนคิดเงิน"
        )

        edited = {}
        cols = st.columns(3)

        for i, name in enumerate(PRICE_MAP):
            edited[name] = cols[i % 3].number_input(
                f"{name} (฿ {PRICE_MAP[name]})",
                min_value=0,
                value=int(counts.get(name, 0)),
                step=1,
                key=f"{file_id}_{name}",
            )


# =====================================================
# คำนวณรายการสินค้า และ โปรโมชั่น
# =====================================================

items = [
    (PRICE_MAP[name], name)
    for name, quantity in edited.items()
    for _ in range(quantity)
]

total_pieces = len(items)

promo = get_promotion(total_pieces)
free_count = promo[1] if promo else 0
free_counts = calculate_free_items(items, free_count)

subtotal = sum(price for price, name in items)
free_value = sum(PRICE_MAP[name] * quantity for name, quantity in free_counts.items())
net = subtotal - free_value
next_promo = get_next_promotion(total_pieces)

# ข้อความชื่อโปรโมชั่นที่จะนำไปแสดงในใบเสร็จ
promo_name_str = f"ซื้อ {promo[0]} แถม {promo[1]}" if promo else "-"

# เตรียมข้อมูลสำหรับใบเสร็จ
rows = []
for name, quantity in edited.items():
    if quantity > 0:
        free_quantity = free_counts.get(name, 0)
        rows.append(
            {
                "รายการ": name,
                "จำนวน": quantity,
                "แถมฟรี": free_quantity,
                "ราคา/ชิ้น": PRICE_MAP[name],
                "ต้องจ่าย": (quantity - free_quantity) * PRICE_MAP[name],
            }
        )


# =====================================================
# ฝั่งขวา
# =====================================================

with right:
    st.subheader("สรุปยอด")

    with st.container(border=True):
        a, b = st.columns(2)

        a.metric("จำนวนทั้งหมด", f"{total_pieces} ชิ้น")
        b.metric("ราคารวม", f"{subtotal} บาท")
        a.metric("ส่วนลดของแถม", f"-{free_value} บาท")
        b.metric("ยอดชำระ", f"{net} บาท")

        # ---------------------------------------------
        # โปรโมชั่นที่ได้รับแล้ว
        # ---------------------------------------------
        if promo:
            st.success(f"🎉 ได้รับโปรโมชั่น ซื้อ {promo[0]} แถม {promo[1]}")
            st.markdown("**🎁 รายการของแถมฟรี**")

            for name, quantity in free_counts.items():
                st.write(
                    f"• {name} × {quantity} "
                    f"(มูลค่า {PRICE_MAP[name] * quantity} บาท)"
                )
        else:
            st.info("ยังไม่เข้าโปรโมชั่น ซื้อครบ 6 ชิ้น รับฟรี 1 ชิ้น")

        # ---------------------------------------------
        # แนะนำโปรโมชั่น
        # ---------------------------------------------
        if next_promo:
            required = next_promo["required"]
            extra = next_promo["extra"]
            pay = next_promo["pay"]
            free = next_promo["free"]
            progress = min(total_pieces / required, 1.0)

            st.markdown("✨ **แนะนำโปรโมชั่นสำหรับคุณ**")

            if extra > 0:
                st.markdown(f"### ซื้อเพิ่มอีก {extra} ชิ้น 🍩")
                st.markdown(
                    f"เพื่อให้ครบ **{required} ชิ้น**\n\n"
                    f"รับโปรโมชั่น **ซื้อ {pay} แถม {free}**\n\n"
                    f"รวมทั้งหมด {required} ชิ้น"
                )
            else:
                st.markdown("### 🎉 เข้าโปรโมชั่นแล้ว!")
                st.markdown(f"ซื้อ {pay} แถม {free}")

            st.progress(progress)
            st.caption(f"ความคืบหน้า {total_pieces}/{required} ชิ้น")
        else:
            st.markdown("✨ **แนะนำโปรโมชั่นสำหรับคุณ**")
            st.markdown("### 🎉 คุณได้รับโปรโมชั่นสูงสุดแล้ว")
            st.caption("ขอบคุณที่เลือก Mister Donut 🍩")
            
    # =================================================
    # แสดงใบเสร็จบน UI ฝั่งขวา 
    # =================================================
    st.subheader("🧾 ใบเสร็จรับเงิน")

    promo_name_str = f"ซื้อ {promo[0]} แถม {promo[1]}" if promo else "-"

    # แยกรายการสินค้าตามโครงสร้างโปรโมชั่น
    if promo:
        pay_req, free_req = promo
        required_pieces = pay_req + free_req

        sorted_items = sorted(items, key=lambda x: x[0])
        free_items = sorted_items[:free_req]
        remaining = sorted_items[free_req:]
        paid_promo_items = remaining[:pay_req]
        excess_items = remaining[pay_req:]  # สินค้าส่วนต่างเกินโปรโมชั่น

        excess_counts = Counter(name for price, name in excess_items)
        paid_promo_counts = Counter(name for price, name in paid_promo_items)
        free_counts_promo = Counter(name for price, name in free_items)
    else:
        pay_req, free_req = 0, 0
        excess_counts = Counter(name for price, name in items)
        paid_promo_counts = Counter()
        free_counts_promo = Counter()

    # สร้าง HTML รายการสินค้าใน TAKE AWAY
    takeaway_content_html = ""

    # 1. สินค้าส่วนต่าง (ถ้ามี)
    if excess_counts:
        for name, qty in excess_counts.items():
            pay_val = qty * PRICE_MAP[name]
            takeaway_content_html += f"""<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="width: 20px;">{qty}</span>
<span style="flex-grow: 1; padding-right: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</span>
<span style="font-weight: bold;">{pay_val:.2f}</span>
</div>"""

    # 2. หัวข้อโปรโมชั่น และ สินค้าจ่ายเงินในโปรโมชั่น
    if promo and paid_promo_counts:
        takeaway_content_html += f"""<div style="font-weight: bold; margin-top: 6px; margin-bottom: 4px;">Promotion {pay_req} Free {free_req}</div>"""
        for name, qty in paid_promo_counts.items():
            pay_val = qty * PRICE_MAP[name]
            takeaway_content_html += f"""<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="width: 20px;">{qty}</span>
<span style="flex-grow: 1; padding-right: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</span>
<span style="font-weight: bold;">{pay_val:.2f}</span>
</div>"""

    # 3. เส้นแบ่ง Free และ รายการสินค้าแถมฟรี
    if promo and free_counts_promo:
        takeaway_content_html += """<div style="text-align: center; font-weight: bold; margin: 6px 0;">----------- Free -----------</div>"""
        for name, qty in free_counts_promo.items():
            takeaway_content_html += f"""<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="width: 20px;">{qty}</span>
<span style="flex-grow: 1; padding-right: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</span>
</div>"""

    # โครงสร้างใบเสร็จ HTML ทั้งหมด
    receipt_html = f"""<div style="background-color: #ffffff; color: #111111; font-family: 'Courier New', Courier, monospace; font-size: 13px; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25); max-width: 360px; margin: 10px auto; line-height: 1.4;">
<div style="text-align: center; font-weight: bold;">
<div>REPRINT CHECK</div>
<div>STORE : 11464</div>
<div style="font-size: 11px;">CENTRAL RESTAURANT GROUP CO.,LTD.</div>
<div style="font-size: 11px;">MISTER DONUT : ROBINSON PRAJEENBURI</div>
<div>TAX INVOICE (ABB.)</div>
<div>Tax ID: 0105532021090</div>
</div>
<div style="border-bottom: 1px dashed #888; margin: 8px 0;"></div>
<div style="display: flex; justify-content: space-between;">
<span>1 POS1</span>
<span>1003 punpan</span>
</div>
<div style="display: flex; justify-content: space-between; font-weight: bold;">
<span>ORDER NO.127</span>
<span>Customer1</span>
</div>
<div style="border-bottom: 1px dashed #888; margin: 8px 0;"></div>

<div style="text-align: center; font-weight: bold; margin-bottom: 8px;">----- TAKE AWAY -----</div>
{takeaway_content_html}

<div style="border-bottom: 1px dashed #888; margin: 8px 0;"></div>
<div style="display: flex; justify-content: space-between;">
<span>Sub Total</span>
<span>{float(subtotal):.2f}</span>
</div>
<div style="display: flex; justify-content: space-between;">
<span>PROMOTION</span>
<span style="font-weight: bold; color: #d9534f;">{promo_name_str}</span>
</div>
<div style="display: flex; justify-content: space-between;">
<span>Discount</span>
<span>{-float(free_value):.2f}</span>
</div>
<div style="display: flex; justify-content: space-between; font-weight: bold;">
<span>Total:</span>
<span>{float(net):.2f} *</span>
</div>
<div style="display: flex; justify-content: space-between; font-weight: bold;">
<span>Payment</span>
<span>{float(net):.2f} *</span>
</div>
<div style="border-bottom: 1px dashed #888; margin: 8px 0;"></div>
<div style="text-align: center; font-weight: bold; margin: 6px 0;">
จำนวนทั้งหมด: {total_pieces} ชิ้น
</div>
<div style="border-bottom: 1px dashed #888; margin: 8px 0;"></div>
<div style="text-align: center; font-size: 11px; color: #444;">
<div>VAT INCLUDED</div>
<div>Thank You</div>
<div>CRG Call Center : 10:00-19:00</div>
<div>TEL. : 1312</div>
<div>callcenter@crg.co.th</div>
</div>
</div>"""

    st.markdown(receipt_html, unsafe_allow_html=True)
