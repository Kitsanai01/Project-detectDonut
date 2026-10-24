import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="Mister Donut Detector", page_icon="🍩", layout="wide")

# =====================================================
# ตั้งค่า
# =====================================================
MODEL_PATH = "best.pt"
IMGSZ = 640            # ต้องตรงกับตอนเทรน (ตัว tuned = 800, ตัวแรก = 640)
IOMA_THRESHOLD = 0.60  # ตัดกรอบที่ซ้อนอยู่ในกรอบอื่น (เพิ่มเป็น 0.70-0.80 ถ้าชิ้นจริงหาย)

# คีย์ต้องตรงกับชื่อคลาสใน data.yaml ทุกตัวอักษร
PRICE_MAP = {
    "Honey Dip": 29,
    "Chocolate ring": 29,
    "Strawberry ring": 29,
    "Choccoconut cake": 29,
    "Coconut cake": 29,
    "Choc peanut cake": 29,
    "Peanut cake": 29,
    "Twist GlaZe": 29,
    "Trio": 29,
    "Blueberry shell": 29,
    "Strawberry shell": 29,
    "Bavarian shell": 29,
    "White Choc PDR": 29,
    "Choc PDR": 29,
    "Glaze PDR": 29,
    "Strawberry PDR": 20,
    "Blueberry PDR": 29,
    "Plain PDR": 29,
}

# (จ่าย, แถม): ซื้อ 5 แถม 1 = 6 ชิ้น | ซื้อ 7 แถม 2 = 9 ชิ้น | ซื้อ 9 แถม 4 = 13 ชิ้น | ซื้อ 10 แถม 5 = 15 ชิ้น
PROMOTION_TIERS = [(5, 1), (7, 2), (9, 4), (10, 5)]


def get_promotion(total):
    """เลือกโปรขั้นสูงสุดที่จำนวนชิ้นถึง"""
    best = None
    for pay, free in PROMOTION_TIERS:
        if total >= pay + free:
            best = (pay, free)
    return best


# =====================================================
# โหลดโมเดล
# =====================================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# =====================================================
# กรองกรอบซ้อน (IoMA = พื้นที่ทับ / พื้นที่กรอบที่เล็กกว่า)
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
# หน้าเว็บ
# =====================================================
st.title("🍩 Mister Donut Detector")

st.sidebar.header("ตั้งค่าการตรวจจับ")
conf = st.sidebar.slider("Confidence", 0.10, 0.90, 0.40, 0.05)
iou = st.sidebar.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05)
ioma = st.sidebar.slider("IoMA (ตัดกรอบซ้อนใน)", 0.30, 0.95, IOMA_THRESHOLD, 0.05)

tab_upload, tab_camera = st.tabs(["📁 อัปโหลดรูป", "📷 ถ่ายรูป"])
with tab_upload:
    up_file = st.file_uploader("อัปโหลดรูปโดนัท", type=["jpg", "jpeg", "png"])
with tab_camera:
    cam_file = st.camera_input("ถ่ายรูปโดนัท")

file = up_file or cam_file

if file:
    # เปิดรูป + หมุนตาม EXIF (รูปจากมือถือมักหมุนผิดถ้าไม่ทำ)
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")

    # สำคัญ: Ultralytics ถือว่า numpy array เป็น BGR จึงต้องสลับจาก RGB ก่อนส่งเข้าโมเดล
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

    # r.plot() คืนค่า BGR จึงสลับเป็น RGB เพื่อแสดงผลใน Streamlit
    plotted_rgb = r.plot()[:, :, ::-1]

    c1, c2 = st.columns(2)
    c1.image(img, caption="ต้นฉบับ", use_container_width=True)
    c2.image(
        plotted_rgb,
        caption=f"ตรวจพบ {len(r.boxes)} ชิ้น",
        use_container_width=True,
    )

    # นับจำนวนแต่ละรสจากผลตรวจจับ
    counts = Counter(r.names[int(b.cls[0])] for b in r.boxes)

    unknown = [n for n in counts if n not in PRICE_MAP]
    if unknown:
        st.warning(f"คลาสที่ยังไม่มีราคาใน PRICE_MAP: {unknown}")

    # -------------------------------------------------
    # ตรวจสอบ/แก้จำนวนก่อนคิดเงิน
    # -------------------------------------------------
    st.subheader("ตรวจสอบ/แก้จำนวนก่อนคิดเงิน")
    st.caption("โมเดลอาจนับผิดรสหรือนับขาด/เกิน แก้จำนวนได้ตรงนี้")

    # ใส่ชื่อไฟล์ใน key เพื่อให้ค่ารีเซ็ตทุกครั้งที่เปลี่ยนรูป
    file_id = getattr(file, "file_id", None) or f"{file.name}_{file.size}"

    edited = {}
    cols = st.columns(3)
    for i, name in enumerate(PRICE_MAP):
        edited[name] = cols[i % 3].number_input(
            name,
            min_value=0,
            value=int(counts.get(name, 0)),
            step=1,
            key=f"{file_id}_{name}",
        )

    # -------------------------------------------------
    # คิดเงิน + โปรโมชั่น (ของแถมคือชิ้นราคาถูกที่สุด)
    # -------------------------------------------------
    items = []
    for name, qty in edited.items():
        items += [PRICE_MAP[name]] * qty

    total_pieces = len(items)
    promo = get_promotion(total_pieces)
    free_count = promo[1] if promo else 0

    items_sorted = sorted(items)
    subtotal = sum(items)
    free_value = sum(items_sorted[:free_count])
    net = subtotal - free_value

    st.subheader("สรุปยอด")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("จำนวนทั้งหมด", f"{total_pieces} ชิ้น")
    m2.metric("ราคารวม", f"{subtotal} บาท")
    m3.metric("ส่วนลดของแถม", f"-{free_value} บาท")
    m4.metric("ยอดชำระ", f"{net} บาท")

    if promo:
        st.success(f"โปรโมชั่น: ซื้อ {promo[0]} แถม {promo[1]}")
        extra = total_pieces - (promo[0] + promo[1])
        if extra > 0:
            st.caption(f"เหลืออีก {extra} ชิ้น คิดราคาปกติ")
    else:
        start = min(p + f for p, f in PROMOTION_TIERS)
        st.info(f"ยังไม่ถึงเงื่อนไขโปรโมชั่น (เริ่มที่ {start} ชิ้น)")

    # -------------------------------------------------
    # ใบเสร็จ (ดาวน์โหลดเป็นข้อความ)
    # -------------------------------------------------
    lines = ["MISTER DONUT RECEIPT", "=" * 40]
    for name, qty in edited.items():
        if qty > 0:
            lines.append(f"{name:<22}{qty:>4} x {PRICE_MAP[name]:>3} = {qty * PRICE_MAP[name]:>5}")
    lines += [
        "-" * 40,
        f"จำนวนทั้งหมด: {total_pieces} ชิ้น",
        f"ราคารวม: {subtotal} บาท",
        f"ส่วนลดของแถม: -{free_value} บาท",
        f"ยอดชำระ: {net} บาท",
    ]
    st.download_button(
        "⬇️ ดาวน์โหลดใบเสร็จ",
        data="\n".join(lines),
        file_name="receipt.txt",
        mime="text/plain",
    )
else:
    st.info("อัปโหลดรูปหรือถ่ายรูปโดนัทเพื่อเริ่มตรวจจับ")