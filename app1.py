
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
# ราคาสินค้า แก้ราคาแต่ละคลาสได้ตรงนี้
# =====================================================
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

# =====================================================
# โปรโมชั่น
# =====================================================
# (จำนวนที่จ่าย, จำนวนที่แถม)
PROMOTION_TIERS = [
    (5, 1),
    (7, 2),
    (9, 4),
    (10, 5),
]


def get_promotion(total):
    """หาสิทธิ์โปรโมชั่นสูงสุดที่จำนวนสินค้าปัจจุบันเข้าเงื่อนไข"""

    best = None

    for pay, free in PROMOTION_TIERS:
        required = pay + free

        if total >= required:
            if best is None or required > best[0] + best[1]:
                best = (pay, free)

    return best


def get_next_promotion(total):
    """หาโปรโมชั่นถัดไปที่ลูกค้ายังต้องซื้อเพิ่ม"""

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
    """
    เลือกของแถมจากสินค้าที่ราคาถูกที่สุด
    ถ้าราคาเท่ากัน ใช้ลำดับรายการใน PRICE_MAP
    """

    sorted_items = sorted(
        items,
        key=lambda x: x[0]
    )

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

    /* กล่องแนะนำโปรโมชั่น */

    .promo-box {
        background: linear-gradient(
            135deg,
            #17334e 0%,
            #10263b 100%
        );

        border: 1px solid #315779;
        border-radius: 14px;

        padding: 18px;
        margin-top: 16px;
        margin-bottom: 12px;

        color: #f5f9ff;
    }

    .promo-title {
        font-size: 17px;
        font-weight: 700;
        color: #80c4ff;
        margin-bottom: 8px;
    }

    .promo-main {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .promo-detail {
        font-size: 14px;
        color: #d4e5f5;
        line-height: 1.7;
    }

    .promo-highlight {
        color: #7ee7c3;
        font-weight: 700;
    }

    .promo-progress {
        background: #29435b;
        border-radius: 20px;
        height: 9px;
        overflow: hidden;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .promo-progress-fill {
        background: linear-gradient(
            90deg,
            #47b5ff,
            #64e3c1
        );

        height: 100%;
        border-radius: 20px;
    }

    .promo-footer {
        font-size: 12px;
        color: #a9c7e2;
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

    area = (
        (xyxy[:, 2] - xyxy[:, 0]) *
        (xyxy[:, 3] - xyxy[:, 1])
    )

    keep = []

    for i in conf.argsort()[::-1]:

        ok = True

        for j in keep:

            ix1 = max(xyxy[i, 0], xyxy[j, 0])
            iy1 = max(xyxy[i, 1], xyxy[j, 1])

            ix2 = min(xyxy[i, 2], xyxy[j, 2])
            iy2 = min(xyxy[i, 3], xyxy[j, 3])

            inter = (
                max(0, ix2 - ix1) *
                max(0, iy2 - iy1)
            )

            smaller = max(
                min(area[i], area[j]),
                1e-6
            )

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

    im = (
        arr_or_img
        if isinstance(arr_or_img, Image.Image)
        else Image.fromarray(arr_or_img)
    )

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

        iou = st.slider(
            "IoU (NMS)",
            0.10,
            0.90,
            0.45,
            0.05
        )

        ioma = st.slider(
            "IoMA (ตัดกรอบซ้อนใน)",
            0.30,
            0.95,
            IOMA_THRESHOLD,
            0.05
        )

    with st.expander("💰 ราคา / โปรโมชั่น"):

        st.caption("ราคาต่อชิ้น")

        for name, price in PRICE_MAP.items():
            st.write(f"• {name}: {price} บาท")

        st.divider()

        for pay, free in PROMOTION_TIERS:

            st.write(
                f"• ซื้อ {pay} แถม {free} "
                f"({pay + free} ชิ้น)"
            )


# =====================================================
# Header + Input
# =====================================================
st.title("🍩 Mister Donut Detector")

st.caption(
    "อัปโหลดหรือถ่ายรูปโดนัท "
    "ระบบจะตรวจจับ นับจำนวน และคำนวณโปรโมชั่นให้"
)

with st.container(border=True):

    tab_upload, tab_camera = st.tabs(
        ["📁 อัปโหลดรูป", "📷 ถ่ายรูป"]
    )

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

    st.info(
        "เริ่มต้นโดยอัปโหลดรูปหรือถ่ายรูปโดนัทด้านบน"
    )

    st.stop()


# =====================================================
# ตรวจจับภาพ
# =====================================================
img = ImageOps.exif_transpose(
    Image.open(file)
).convert("RGB")

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

counts = Counter(
    r.names[int(b.cls[0])]
    for b in r.boxes
)


unknown = [
    name
    for name in counts
    if name not in PRICE_MAP
]

if unknown:

    st.warning(
        f"คลาสที่ยังไม่มีราคาใน PRICE_MAP: {unknown}"
    )


file_id = (
    getattr(file, "file_id", None)
    or f"{file.name}_{file.size}"
)


# =====================================================
# Layout หลัก
# =====================================================
left, right = st.columns(
    [3, 2],
    gap="large"
)


# =====================================================
# ฝั่งซ้าย: รูปและแก้จำนวน
# =====================================================
with left:

    st.subheader("ผลการตรวจจับ")

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            to_display(img),
            caption="ต้นฉบับ"
        )

    with c2:

        st.image(
            to_display(plotted_rgb),
            caption=f"ตรวจพบ {len(r.boxes)} ชิ้น"
        )


    with st.expander(
        "✏️ ตรวจสอบ/แก้จำนวนแต่ละรส",
        expanded=True
    ):

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
# คำนวณยอด
# =====================================================

# รายการสินค้าทั้งหมด
items = [
    (PRICE_MAP[name], name)
    for name, quantity in edited.items()
    for _ in range(quantity)
]

total_pieces = len(items)

# โปรโมชั่นที่ได้รับแล้ว
promo = get_promotion(total_pieces)

free_count = promo[1] if promo else 0

# เลือกสินค้าราคาถูกที่สุดเป็นของแถม
free_counts = calculate_free_items(
    items,
    free_count
)

# ราคารวมก่อนหักส่วนลด
subtotal = sum(
    price
    for price, name in items
)

# มูลค่าของแถม
free_value = sum(
    PRICE_MAP[name] * quantity
    for name, quantity in free_counts.items()
)

# ยอดชำระ
net = subtotal - free_value


# =====================================================
# คำนวณโปรโมชั่นถัดไป
# =====================================================
next_promo = get_next_promotion(total_pieces)


# =====================================================
# ฝั่งขวา: สรุปยอด
# =====================================================
with right:

    st.subheader("สรุปยอด")

    with st.container(border=True):

        a, b = st.columns(2)

        a.metric(
            "จำนวนทั้งหมด",
            f"{total_pieces} ชิ้น"
        )

        b.metric(
            "ราคารวม",
            f"{subtotal} บาท"
        )

        a.metric(
            "ส่วนลดของแถม",
            f"-{free_value} บาท"
        )

        b.metric(
            "ยอดชำระ",
            f"{net} บาท"
        )


        # ---------------------------------------------
        # โปรโมชั่นที่ได้รับแล้ว
        # ---------------------------------------------
        if promo:

            st.success(
                f"🎉 ได้รับโปรโมชั่น "
                f"ซื้อ {promo[0]} แถม {promo[1]}"
            )

            st.markdown("**🎁 รายการของแถมฟรี**")

            for name, quantity in free_counts.items():

                st.write(
                    f"• {name} × {quantity} "
                    f"(มูลค่า {PRICE_MAP[name] * quantity} บาท)"
                )

        else:

            st.info(
                "ยังไม่เข้าโปรโมชั่น "
                "ซื้อครบ 6 ชิ้น รับฟรี 1 ชิ้น"
            )


        # ---------------------------------------------
        # กล่องแนะนำโปรโมชั่นสีฟ้า
        # ---------------------------------------------
        st.markdown(
            '<div class="promo-box">',
            unsafe_allow_html=True
        )

        if next_promo:

            required = next_promo["required"]
            extra = next_promo["extra"]

            pay = next_promo["pay"]
            free = next_promo["free"]

            progress = min(
                total_pieces / required * 100,
                100
            )

            st.markdown(
                """
                <div class="promo-title">
                    ✨ แนะนำโปรโมชั่นสำหรับคุณ
                </div>
                """,
                unsafe_allow_html=True
            )

            if extra > 0:

                st.markdown(
                    f"""
                    <div class="promo-main">
                        ซื้อเพิ่มอีก {extra} ชิ้น 🍩
                    </div>

                    <div class="promo-detail">
                        เพื่อให้ครบ <b>{required} ชิ้น</b>
                        <br>
                        รับโปรโมชั่น
                        <span class="promo-highlight">
                            ซื้อ {pay} แถม {free}
                        </span>
                        <br>
                        รวมทั้งหมด {required} ชิ้น
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="promo-main">
                        🎉 เข้าโปรโมชั่นแล้ว!
                    </div>

                    <div class="promo-detail">
                        ซื้อ {pay} แถม {free}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown(
                f"""
                <div class="promo-progress">
                    <div
                        class="promo-progress-fill"
                        style="width: {progress}%"
                    ></div>
                </div>

                <div class="promo-footer">
                    ความคืบหน้า {total_pieces}/{required} ชิ้น
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="promo-title">
                    🎉 คุณได้รับโปรโมชั่นสูงสุดแล้ว
                </div>

                <div class="promo-detail">
                    ขอบคุณที่เลือก Mister Donut 🍩
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =================================================
    # รายการสินค้า
    # =================================================
    st.markdown("**รายการที่เลือก**")

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
                    "ต้องจ่าย": (
                        quantity - free_quantity
                    ) * PRICE_MAP[name],
                }
            )


    if rows:

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True
        )

    else:

        st.caption("ยังไม่มีรายการ")


    # =================================================
    # ใบเสร็จ
    # =================================================
    lines = [
        "MISTER DONUT RECEIPT",
        "=" * 40
    ]

    for row in rows:

        tag = (
            f"  (แถม {row['แถมฟรี']})"
            if row["แถมฟรี"]
            else ""
        )

        lines.append(
            f"{row['รายการ']:<20}"
            f"{row['จำนวน']:>3} x "
            f"{row['ราคา/ชิ้น']:>3} = "
            f"{row['ต้องจ่าย']:>5}"
            f"{tag}"
        )


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
        use_container_width=True,
        disabled=not rows,
    )
