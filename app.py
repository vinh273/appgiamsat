import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Cấu hình tiêu đề trang web
st.set_page_config(page_title="AIoT - Nhận Diện Bệnh Lúa", page_icon="🌾", layout="wide")

# 1. Cơ sở dữ liệu nông nghiệp: Chi tiết bệnh và biện pháp phòng trừ
db_benh_lua = {
    "da_on": {
        "ten_benh": "Bệnh Đạo ôn (Pyricularia oryzae)",
        "trieu_chung": "Vết bệnh hình thoi, tâm màu xám trắng, viền nâu đậm trên lá.",
        "nguyen_nhan": "Do nấm bệnh phát triển mạnh trong điều kiện độ ẩm cao, sương mù, bón thừa phân đạm.",
        "phong_tri": [
            "Giữ mực nước ruộng ổn định, tránh để ruộng khô hạn xen kẽ.",
            "Ngưng ngay việc bón phân đạm và phân bón lá khi phát hiện bệnh chớm xuất hiện.",
            "Phun các thuốc đặc trị chứa hoạt chất: Tricyclazole, Isoprothiolane hoặc Fenoxanil."
        ]
    },
    "kho_van": {
        "ten_benh": "Bệnh Khô vằn (Rhizoctonia solani)",
        "trieu_chung": "Vết bệnh loang lổ dạng vằn mây, xuất hiện nhiều ở bẹ lá gần mặt nước.",
        "nguyen_nhan": "Do gieo sạ quá dày, ruộng rậm rạp, thời tiết nóng ẩm.",
        "phong_tri": [
            "Tỉa bớt lá chân, rút bớt nước trong ruộng để hạ độ ẩm không khí tầng thấp.",
            "Cân đối lại dinh dưỡng, tránh bón thừa đạm.",
            "Sử dụng các loại thuốc trừ nấm có hoạt chất: Validamycin, Hexaconazole hoặc Jinggangmycin."
        ]
    },
    "bac_la": {
        "ten_benh": "Bệnh Bạc lá / Cháy bìa lá (Xanthomonas oryzae)",
        "trieu_chung": "Vết bệnh khô cháy từ chóp lá lan dần xuống dưới theo hai bên mép lá.",
        "nguyen_nhan": "Do vi khuẩn tấn công qua các vết thương cơ học sau mưa bão, giông lốc.",
        "phong_tri": [
            "Khơi thông dòng chảy, thay nước ruộng bị ô nhiễm, giữ ruộng thông thoáng.",
            "Tuyệt đối dừng bón đạm và phân bón qua lá.",
            "Phun thuốc đặc trị vi khuẩn chứa hoạt chất: Bismerthiazol, Copper hydroxide hoặc Kasugamycin."
        ]
    }
}

# 2. Tải mô hình AI YOLO (đảm bảo file best.pt nằm cùng thư mục với app.py)
@st.cache_resource
def load_yolo_model():
    model_path = "best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        return None

model = load_yolo_model()

# 3. Giao diện trang web chính
st.title("🌾 Hệ Thống AIoT & UAV: Nhận Diện Bệnh Lúa & Tư Vấn Phòng Trị")
st.markdown("---")

# Kiểm tra xem đã có file best.pt chưa
if model is None:
    st.error("⚠️ Không tìm thấy file trọng số **'best.pt'**! Vui lòng copy file `best.pt` vào chung thư mục với file `app.py`.")
else:
    # Khu vực tải ảnh lên
    uploaded_file = st.file_uploader("📤 Tải lên hình ảnh lá lúa cần chẩn đoán ( định dạng .jpg, .png ):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("📷 Ảnh gốc lá lúa")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("🔍 Kết quả chẩn đoán từ AI")
            if st.button("Bắt đầu phân tích bệnh", type="primary"):
                with st.spinner("Hệ thống AI đang xử lý và phân tích vết bệnh..."):
                    # Lưu tạm ảnh để mô hình dự đoán
                    temp_path = "temp_uploaded_image.jpg"
                    image.save(temp_path)
                    
                    # Chạy dự đoán
                    results = model(temp_path)
                    
                    # Trích xuất kết quả (đối với mô hình Classify)
                    try:
                        predicted_class_id = results[0].probs.top1
                        predicted_class_name = results[0].names[predicted_class_id]
                        confidence = float(results[0].probs.top1conf) * 100
                        
                        st.success(f"**Tên bệnh dự đoán:** {predicted_class_name}")
                        st.info(f"**Độ tin cậy của mô hình:** {confidence:.2f}%")
                        
                        # Hiển thị thông tin phòng trị tương ứng
                        if predicted_class_name in db_benh_lua:
                            info = db_benh_lua[predicted_class_name]
                            st.markdown(f"### 📋 Chi tiết bệnh học:")
                            st.write(f"- **Tên khoa học:** {info['ten_benh']}")
                            st.write(f"- **Triệu chứng:** {info['trieu_chung']}")
                            st.write(f"- **Nguyên nhân:** {info['nguyen_nhan']}")
                            
                            st.markdown(f"### 🛡️ Biện pháp phòng trừ khuyến cáo:")
                            for idx, step in enumerate(info['phong_tri'], 1):
                                st.write(f"{idx}. {step}")
                        else:
                            st.warning("⚠️ Nhận diện được phân loại nhưng chưa có dữ liệu chi tiết cho loại bệnh này trong hệ thống.")
                            
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi trong quá trình xử lý kết quả: {e}")