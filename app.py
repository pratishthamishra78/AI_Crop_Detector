import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Crop Doctor",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
    }

    .small-text {
        color: #666;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_crop_model():
    return load_model("crop_disease_model.h5")

@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_crop_model()
labels = load_labels()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🌱 AI Crop Doctor")

st.sidebar.markdown("""
### Navigation

🔍 **Disease Detection**  
💊 **Treatment & Prevention**  
🤖 **Farmer Assistant**  
👨‍🌾 **Contact Expert**
""")

st.sidebar.divider()

st.sidebar.info(
    "Upload a clear image of a crop leaf and let the AI model analyze it."
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🌱 AI Crop Doctor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered crop disease detection and farmer assistance'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Disease Detection",
    "💊 Treatment & Prevention",
    "🤖 Farmer Assistant",
    "👨‍🌾 Contact Expert"
])

# ==================================================
# TAB 1 - DISEASE DETECTION
# ==================================================
with tab1:

    st.subheader("📷 Upload Crop Leaf")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the affected crop leaf."
    )

    if uploaded_file:

        try:
            # Open image
            img = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    img,
                    caption="🖼️ Uploaded Leaf Image",
                    use_container_width=True
                )

            # --------------------------------------------------
            # PREPROCESS IMAGE
            # --------------------------------------------------
            resized_img = img.resize((224, 224))

            img_array = image.img_to_array(resized_img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------
            with st.spinner("🔎 Analyzing your crop..."):
                pred = model.predict(img_array, verbose=0)

            predicted_index = int(np.argmax(pred))
            predicted_disease = labels[predicted_index]
            confidence = float(np.max(pred)) * 100

            # --------------------------------------------------
            # SAVE PREDICTION FOR OTHER TABS
            # --------------------------------------------------
            st.session_state["predicted_disease"] = predicted_disease
            st.session_state["confidence"] = confidence

            # --------------------------------------------------
            # DISPLAY RESULT
            # --------------------------------------------------
            with col2:

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.subheader("🔬 AI Diagnosis")

                st.success(
                    f"🌾 **{predicted_disease}**"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                if confidence >= 80:
                    st.success("🟢 High confidence prediction")

                elif confidence >= 50:
                    st.warning("🟡 Moderate confidence prediction")

                else:
                    st.error("🔴 Low confidence prediction")

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            st.divider()

            st.warning(
                "⚠️ AI predictions are for assistance only. "
                "For serious crop damage, confirm the diagnosis with "
                "an agriculture expert before taking treatment decisions."
            )

        except Exception as e:

            st.error(
                f"❌ Error processing image: {e}"
            )

    else:

        st.info(
            "👆 Upload a crop leaf image above to start disease detection."
        )


# ==================================================
# TAB 2 - TREATMENT & PREVENTION
# ==================================================
with tab2:

    st.subheader("💊 Treatment & Prevention")

    st.write(
        "After identifying a disease, follow these general management steps. "
        "Treatment should be confirmed according to the crop, region and severity."
    )

    # ==================================================
    # 38 DISEASE INFORMATION
    # ==================================================
    disease_info = {

    # ==================== APPLE ====================

    "Apple___Apple_scab": {
        "title": "🍎 Apple Scab",
        "symptoms": [
            "Leaves par olive/green se brown circular spots appear ho sakte hain.",
            "Severe infection me leaves yellow hokar gir sakte hain.",
            "Fruit par rough, dark scab-like lesions develop ho sakte hain."
        ],
        "treatment": [
            "Infected leaves aur fallen plant debris ko collect karke properly dispose karein.",
            "Canopy ko open rakhein taaki leaves jaldi dry ho sakein.",
            "Overhead irrigation aur unnecessary leaf wetness avoid karein.",
            "Infection ko regularly monitor karein, especially humid weather me.",
            "Disease pressure zyada ho to local agricultural expert se crop-stage ke according approved fungicide options ke baare me advice lein."
        ],
        "prevention": [
            "Orchard sanitation maintain karein.",
            "Fallen infected leaves ko remove karein.",
            "Good airflow aur proper pruning maintain karein.",
            "Regular monitoring early stage par karein."
        ]
    },

    "Apple___Black_rot": {
        "title": "🍎 Apple Black Rot",
        "symptoms": [
            "Leaves par purple/brown circular spots develop ho sakte hain.",
            "Fruit par brown/black rotting area develop ho sakta hai.",
            "Branches par dark/cankered areas appear ho sakte hain."
        ],
        "treatment": [
            "Affected fruits ko plant se remove karein.",
            "Dead ya diseased branches ko prune karein.",
            "Pruning tools ko plants ke beech clean/disinfect karein.",
            "Orchard me fallen fruits aur plant debris remove karein.",
            "Severe infection me local expert se approved disease-management treatment consult karein."
        ],
        "prevention": [
            "Orchard sanitation maintain karein.",
            "Dead wood aur mummified fruits remove karein.",
            "Plant injuries ko minimize karein.",
            "Regularly branches aur fruits inspect karein."
        ]
    },

    "Apple___Cedar_apple_rust": {
        "title": "🍎 Apple Cedar Apple Rust",
        "symptoms": [
            "Leaves par yellow/orange spots appear ho sakte hain.",
            "Spots ke neeche orange-yellow fungal structures develop ho sakte hain.",
            "Severe infection premature leaf drop cause kar sakti hai."
        ],
        "treatment": [
            "Affected plant material ko monitor aur remove karein where practical.",
            "Orchard me good airflow maintain karein.",
            "Wet foliage ko unnecessarily prolonged na hone dein.",
            "Nearby susceptible host plants ki presence ko bhi consider karein.",
            "Fungicide decision crop stage aur local recommendations ke according agricultural expert se confirm karein."
        ],
        "prevention": [
            "Regular scouting karein.",
            "Orchard sanitation maintain karein.",
            "Good canopy airflow maintain karein.",
            "Disease-prone conditions me preventive management ke liye expert advice lein."
        ]
    },

    "Apple___healthy": {
        "title": "🍎 Apple — Healthy Crop",
        "symptoms": [
            "Disease ke obvious symptoms detect nahi hue.",
            "Leaves generally healthy appearance show kar rahe hain."
        ],
        "treatment": [
            "Disease treatment ki currently zarurat nahi hai.",
            "Plant ko balanced irrigation aur nutrition provide karein.",
            "Leaves aur fruits ko regularly monitor karte rahein."
        ],
        "prevention": [
            "Orchard sanitation maintain karein.",
            "Good airflow maintain karein.",
            "Regular disease scouting karein."
        ]
    },


    # ==================== BLUEBERRY ====================

    "Blueberry___healthy": {
        "title": "🫐 Blueberry — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue.",
            "Leaves healthy appearance show kar rahe hain."
        ],
        "treatment": [
            "Specific disease treatment ki zarurat nahi hai.",
            "Proper irrigation aur nutrition maintain karein.",
            "Plants ko regularly inspect karein."
        ],
        "prevention": [
            "Good airflow maintain karein.",
            "Excessive moisture avoid karein.",
            "Diseased plant material ko promptly remove karein if detected."
        ]
    },


    # ==================== CHERRY ====================

    "Cherry_(including_sour)___Powdery_mildew": {
        "title": "🍒 Cherry Powdery Mildew",
        "symptoms": [
            "Leaves ya shoots par white powder-like fungal growth appear ho sakti hai.",
            "Young leaves distort ya curl ho sakti hain.",
            "Growth aur fruit development affect ho sakta hai."
        ],
        "treatment": [
            "Heavily affected plant parts ko remove karein where practical.",
            "Canopy me airflow improve karein.",
            "Excessive nitrogen application avoid karein.",
            "Plant ko regularly monitor karein.",
            "Disease severe ho to locally approved fungicide options ke liye agricultural expert se consult karein."
        ],
        "prevention": [
            "Proper pruning aur airflow maintain karein.",
            "Excessive humidity aur dense canopy avoid karein.",
            "Early symptoms par monitoring start karein."
        ]
    },

    "Cherry_(including_sour)___healthy": {
        "title": "🍒 Cherry — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal irrigation, nutrition aur crop monitoring continue karein."
        ],
        "prevention": [
            "Good airflow maintain karein.",
            "Orchard sanitation maintain karein.",
            "Regularly leaves aur fruits inspect karein."
        ]
    },


    # ==================== CORN ====================

    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "title": "🌽 Corn Cercospora / Gray Leaf Spot",
        "symptoms": [
            "Leaves par elongated gray/brown lesions develop ho sakte hain.",
            "Lesions leaf veins ke parallel appear ho sakte hain.",
            "Severe infection me photosynthetic leaf area reduce ho sakta hai."
        ],
        "treatment": [
            "Affected field ko regularly scout karein.",
            "Crop residue management par attention dein.",
            "Excessive leaf wetness aur poor airflow conditions minimize karein.",
            "Severe disease pressure me locally recommended management options ke liye agricultural expert se consult karein."
        ],
        "prevention": [
            "Crop rotation consider karein.",
            "Resistant/tolerant varieties where available use karein.",
            "Field sanitation maintain karein.",
            "Early disease monitoring karein."
        ]
    },

    "Corn___Common_rust_": {
        "title": "🌽 Corn Common Rust",
        "symptoms": [
            "Leaves par reddish-brown/rust-colored pustules appear ho sakte hain.",
            "Severe infection me leaves ki photosynthetic capacity reduce ho sakti hai."
        ],
        "treatment": [
            "Plants ko regularly inspect karein.",
            "Severely affected crop areas ko closely monitor karein.",
            "Unnecessary prolonged leaf wetness avoid karein.",
            "Disease pressure aur crop stage ke basis par treatment decision agricultural expert se confirm karein."
        ],
        "prevention": [
            "Resistant varieties use karein where available.",
            "Regular scouting karein.",
            "Field conditions aur crop health maintain karein."
        ]
    },

    "Corn___Northern_Leaf_Blight": {
        "title": "🌽 Corn Northern Leaf Blight",
        "symptoms": [
            "Leaves par long, cigar-shaped gray-green/brown lesions develop ho sakte hain.",
            "Severe infection me large portions of leaf tissue affected ho sakte hain."
        ],
        "treatment": [
            "Affected field ko frequently monitor karein.",
            "Crop residue management improve karein.",
            "Good crop health maintain karein.",
            "Severe disease pressure me approved fungicide options ke liye local expert se advice lein."
        ],
        "prevention": [
            "Crop rotation consider karein.",
            "Resistant/tolerant varieties use karein.",
            "Crop residue management maintain karein.",
            "Early scouting karein."
        ]
    },

    "Corn___healthy": {
        "title": "🌽 Corn — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal irrigation, nutrition aur crop monitoring continue karein."
        ],
        "prevention": [
            "Regular field scouting karein.",
            "Balanced crop nutrition maintain karein.",
            "Field sanitation aur proper crop management follow karein."
        ]
    },


    # ==================== GRAPE ====================

    "Grape___Black_rot": {
        "title": "🍇 Grape Black Rot",
        "symptoms": [
            "Leaves par tan/brown circular spots appear ho sakte hain.",
            "Fruit par dark lesions develop hokar fruit shrivel ho sakta hai.",
            "Affected berries eventually become dark and mummified."
        ],
        "treatment": [
            "Infected berries aur mummified fruit remove karein.",
            "Diseased leaves aur plant debris remove karein.",
            "Canopy airflow improve karein.",
            "Excessive leaf wetness avoid karein.",
            "Severe infection me local agricultural expert se approved fungicide management ke baare me consult karein."
        ],
        "prevention": [
            "Vineyard sanitation maintain karein.",
            "Mummified berries remove karein.",
            "Proper pruning aur canopy management karein.",
            "Regular scouting karein."
        ]
    },

    "Grape___Esca_(Black_Measles)": {
        "title": "🍇 Grape Esca (Black Measles)",
        "symptoms": [
            "Leaves par interveinal yellowing aur necrotic patterns appear ho sakte hain.",
            "Fruit par dark spots develop ho sakte hain.",
            "Disease vine health ko gradually reduce kar sakti hai."
        ],
        "treatment": [
            "Affected vines ko carefully monitor karein.",
            "Dead/diseased wood ko remove karne ke liye proper pruning practices follow karein.",
            "Pruning wounds ko minimize karein.",
            "Severely affected vines ke management ke liye vineyard expert se diagnosis confirm karein."
        ],
        "prevention": [
            "Healthy planting material use karein.",
            "Pruning practices carefully manage karein.",
            "Vineyard sanitation maintain karein.",
            "Disease symptoms early stage par identify karein."
        ]
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "title": "🍇 Grape Leaf Blight",
        "symptoms": [
            "Leaves par brown/dark spots develop ho sakte hain.",
            "Spots expand hokar larger damaged areas bana sakte hain.",
            "Severe infection me premature leaf drop ho sakta hai."
        ],
        "treatment": [
            "Affected leaves ko monitor aur remove karein where practical.",
            "Canopy ventilation improve karein.",
            "Excessive moisture avoid karein.",
            "Fallen infected plant material remove karein.",
            "Severe disease me locally approved management options ke liye expert se consult karein."
        ],
        "prevention": [
            "Good canopy management karein.",
            "Vineyard sanitation maintain karein.",
            "Regular scouting karein."
        ]
    },

    "Grape___healthy": {
        "title": "🍇 Grape — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal vineyard management continue karein."
        ],
        "prevention": [
            "Canopy airflow maintain karein.",
            "Vineyard sanitation maintain karein.",
            "Regular leaf aur fruit inspection karein."
        ]
    },


    # ==================== ORANGE ====================

    "Orange___Haunglongbing_(Citrus_greening)": {
        "title": "🍊 Citrus Greening / Huanglongbing (HLB)",
        "symptoms": [
            "Leaves par uneven yellowing ya blotchy mottling appear ho sakti hai.",
            "New shoots weak aur yellow ho sakte hain.",
            "Fruit development aur quality decline ho sakti hai."
        ],
        "treatment": [
            "Suspected infection ko agricultural expert se confirm karayein.",
            "Infected tree ko closely monitor karein.",
            "Citrus psyllid/vector management ko local recommendations ke according follow karein.",
            "Severely affected trees ke removal/management ka decision local plant-health authority ya expert ke guidance se lein.",
            "Unverified home remedies ya random pesticide applications avoid karein."
        ],
        "prevention": [
            "Certified healthy planting material use karein.",
            "Vector monitoring aur management karein.",
            "Orchard me regular scouting karein.",
            "Suspected cases ko promptly report/consult karein."
        ]
    },


    # ==================== PEACH ====================

    "Peach___Bacterial_spot": {
        "title": "🍑 Peach Bacterial Spot",
        "symptoms": [
            "Leaves par small dark spots develop ho sakte hain.",
            "Fruit par sunken spots/lesions appear ho sakte hain.",
            "Severe infection me premature leaf drop ho sakta hai."
        ],
        "treatment": [
            "Affected plant material ko remove where practical.",
            "Overhead irrigation avoid karein.",
            "Canopy airflow maintain karein.",
            "Plant stress ko minimize karein.",
            "Bacterial disease management ke liye locally approved options agricultural expert se confirm karein."
        ],
        "prevention": [
            "Healthy planting material use karein.",
            "Good orchard sanitation maintain karein.",
            "Excessive leaf wetness avoid karein.",
            "Regular monitoring karein."
        ]
    },

    "Peach___healthy": {
        "title": "🍑 Peach — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal orchard management continue karein."
        ],
        "prevention": [
            "Regular scouting karein.",
            "Good airflow aur sanitation maintain karein."
        ]
    },


    # ==================== PEPPER ====================

    "Pepper,_bell___Bacterial_spot": {
        "title": "🌶️ Bell Pepper Bacterial Spot",
        "symptoms": [
            "Leaves par small dark spots appear ho sakte hain.",
            "Fruit par raised/scabby spots develop ho sakte hain.",
            "Severe infection plant vigor reduce kar sakti hai."
        ],
        "treatment": [
            "Severely affected plant material remove karein where practical.",
            "Overhead watering avoid karein.",
            "Wet foliage ko prolonged duration tak na rehne dein.",
            "Field sanitation maintain karein.",
            "Disease severe ho to local agricultural expert se approved management options confirm karein."
        ],
        "prevention": [
            "Disease-free seeds/transplants use karein.",
            "Crop rotation follow karein.",
            "Tools aur hands ko clean rakhein.",
            "Regular scouting karein."
        ]
    },

    "Pepper,_bell___healthy": {
        "title": "🌶️ Bell Pepper — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal irrigation aur crop care continue karein."
        ],
        "prevention": [
            "Regular crop monitoring karein.",
            "Good airflow maintain karein.",
            "Overwatering avoid karein."
        ]
    },


    # ==================== POTATO ====================

    "Potato___Early_blight": {
        "title": "🥔 Potato Early Blight",
        "symptoms": [
            "Older leaves par brown circular lesions appear ho sakte hain.",
            "Lesions me concentric ring pattern ho sakta hai.",
            "Severe infection me leaf yellowing aur defoliation ho sakti hai."
        ],
        "treatment": [
            "Affected leaves ko monitor karein aur severely damaged tissue remove where practical.",
            "Plant stress aur nutrient imbalance ko minimize karein.",
            "Overhead irrigation avoid karein.",
            "Field sanitation maintain karein.",
            "Disease pressure high ho to locally approved fungicide program ke liye agricultural expert se consult karein."
        ],
        "prevention": [
            "Crop rotation follow karein.",
            "Healthy planting material use karein.",
            "Crop debris management karein.",
            "Regular scouting karein."
        ]
    },

    "Potato___Late_blight": {
        "title": "🥔 Potato Late Blight",
        "symptoms": [
            "Leaves par dark water-soaked lesions appear ho sakte hain.",
            "Cool/humid conditions me disease rapidly spread kar sakti hai.",
            "Tubers bhi affected ho sakte hain."
        ],
        "treatment": [
            "Suspected infection ko urgently monitor karein.",
            "Affected plant material ko manage/remove according to local guidance.",
            "Field me prolonged leaf wetness avoid karein.",
            "Healthy plants ko closely monitor karein.",
            "Late blight rapidly spread kar sakta hai, isliye locally recommended fungicide management ke liye agricultural expert se promptly consult karein."
        ],
        "prevention": [
            "Disease-free seed potatoes use karein.",
            "Crop rotation follow karein.",
            "Good field drainage maintain karein.",
            "Weather conditions aur crop symptoms regularly monitor karein."
        ]
    },

    "Potato___healthy": {
        "title": "🥔 Potato — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal crop management continue karein."
        ],
        "prevention": [
            "Healthy seed material use karein.",
            "Regular field scouting karein.",
            "Proper irrigation aur drainage maintain karein."
        ]
    },


    # ==================== RASPBERRY ====================

    "Raspberry___healthy": {
        "title": "🫐 Raspberry — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal irrigation, nutrition aur crop monitoring continue karein."
        ],
        "prevention": [
            "Good airflow maintain karein.",
            "Plant debris remove karein.",
            "Regularly leaves aur fruits inspect karein."
        ]
    },


    # ==================== SOYBEAN ====================

    "Soybean___healthy": {
        "title": "🌱 Soybean — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal crop management continue karein."
        ],
        "prevention": [
            "Regular field scouting karein.",
            "Balanced nutrition maintain karein.",
            "Proper irrigation management karein."
        ]
    },


    # ==================== SQUASH ====================

    "Squash___Powdery_mildew": {
        "title": "🎃 Squash Powdery Mildew",
        "symptoms": [
            "Leaves par white powder-like fungal growth appear hoti hai.",
            "Leaves yellow ya dry ho sakti hain.",
            "Severe infection plant vigor reduce kar sakti hai."
        ],
        "treatment": [
            "Severely affected leaves ko remove where practical.",
            "Plant ke around airflow improve karein.",
            "Overhead watering avoid karein.",
            "Dense growth ko manage karein.",
            "Severe infection me locally approved fungicide options ke liye agricultural expert se consult karein."
        ],
        "prevention": [
            "Good spacing maintain karein.",
            "Air circulation improve karein.",
            "Regular scouting karein.",
            "Plant stress minimize karein."
        ]
    },


    # ==================== STRAWBERRY ====================

    "Strawberry___Leaf_scorch": {
        "title": "🍓 Strawberry Leaf Scorch",
        "symptoms": [
            "Leaves par dark purple/brown spots develop ho sakte hain.",
            "Spots expand hokar leaf tissue ko damage kar sakte hain.",
            "Severe infection me leaves scorched appearance de sakti hain."
        ],
        "treatment": [
            "Severely affected leaves remove karein where practical.",
            "Plant debris remove karein.",
            "Good airflow maintain karein.",
            "Excessive moisture avoid karein.",
            "Disease severe ho to local expert se management options confirm karein."
        ],
        "prevention": [
            "Field sanitation maintain karein.",
            "Healthy planting material use karein.",
            "Regular monitoring karein."
        ]
    },

    "Strawberry___healthy": {
        "title": "🍓 Strawberry — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal crop care continue karein."
        ],
        "prevention": [
            "Good airflow maintain karein.",
            "Overwatering avoid karein.",
            "Regularly leaves aur fruits inspect karein."
        ]
    },


    # ==================== TOMATO ====================

    "Tomato___Bacterial_spot": {
        "title": "🍅 Tomato Bacterial Spot",
        "symptoms": [
            "Leaves par small dark spots appear ho sakte hain.",
            "Fruit par small raised/dark lesions develop ho sakte hain.",
            "Severe infection me leaf drop ho sakta hai."
        ],
        "treatment": [
            "Severely infected leaves/plant parts remove where practical.",
            "Overhead irrigation avoid karein.",
            "Wet foliage ko dry rehne ka sufficient time dein.",
            "Tools ko plants ke beech clean karein.",
            "Severe disease me locally approved bacterial disease management ke liye agricultural expert se consult karein."
        ],
        "prevention": [
            "Disease-free seeds/transplants use karein.",
            "Crop rotation follow karein.",
            "Field sanitation maintain karein.",
            "Regular scouting karein."
        ]
    },

    "Tomato___Early_blight": {
        "title": "🍅 Tomato Early Blight",
        "symptoms": [
            "Older leaves par brown circular lesions appear ho sakte hain.",
            "Lesions me concentric rings ho sakti hain.",
            "Leaves yellow hokar prematurely fall kar sakti hain."
        ],
        "treatment": [
            "Affected leaves ko remove karein where practical.",
            "Plant debris ko field se remove karein.",
            "Overhead watering avoid karein.",
            "Plants ke beech sufficient airflow maintain karein.",
            "Disease pressure high ho to locally approved fungicide options ke liye agricultural expert se consult karein."
        ],
        "prevention": [
            "Crop rotation follow karein.",
            "Mulching/soil splash reduction consider karein.",
            "Healthy plant nutrition maintain karein.",
            "Regular scouting karein."
        ]
    },

    "Tomato___Late_blight": {
        "title": "🍅 Tomato Late Blight",
        "symptoms": [
            "Leaves par dark water-soaked lesions develop ho sakte hain.",
            "Lesions rapidly expand ho sakte hain.",
            "Humid conditions me disease quickly spread kar sakti hai."
        ],
        "treatment": [
            "Suspected infection ko immediately monitor karein.",
            "Heavily affected plant material ko manage/remove according to local guidance.",
            "Overhead irrigation avoid karein.",
            "Healthy plants ko closely inspect karein.",
            "Rapid spread possible hone ke karan agricultural expert se prompt management advice lein."
        ],
        "prevention": [
            "Good airflow maintain karein.",
            "Prolonged leaf wetness avoid karein.",
            "Affected plant debris remove karein.",
            "Regular weather-based disease monitoring karein."
        ]
    },

    "Tomato___Leaf_Mold": {
        "title": "🍅 Tomato Leaf Mold",
        "symptoms": [
            "Upper leaf surface par yellow patches appear ho sakte hain.",
            "Leaf ke underside par olive/grayish mold growth ho sakti hai.",
            "High humidity disease development ko favor kar sakti hai."
        ],
        "treatment": [
            "Affected leaves remove karein where practical.",
            "Greenhouse/field ventilation improve karein.",
            "Humidity aur leaf wetness reduce karein.",
            "Overcrowding avoid karein.",
            "Severe disease me locally approved treatment options ke liye expert se consult karein."
        ],
        "prevention": [
            "Good ventilation maintain karein.",
            "Overhead irrigation avoid karein.",
            "Plant spacing maintain karein.",
            "Regularly inspect leaf undersides."
        ]
    },

    "Tomato___Septoria_leaf_spot": {
        "title": "🍅 Tomato Septoria Leaf Spot",
        "symptoms": [
            "Leaves par numerous small circular spots develop ho sakte hain.",
            "Spots ke centers gray/tan aur margins dark ho sakte hain.",
            "Lower leaves usually pehle affected ho sakti hain."
        ],
        "treatment": [
            "Affected lower leaves remove karein where practical.",
            "Plant debris remove karein.",
            "Soil splash aur overhead irrigation reduce karein.",
            "Good airflow maintain karein.",
            "Severe infection me approved fungicide management ke liye local expert se consult karein."
        ],
        "prevention": [
            "Crop rotation follow karein.",
            "Mulch use karke soil splash reduce karein.",
            "Healthy planting material use karein.",
            "Regular scouting karein."
        ]
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "🍅 Tomato Two-Spotted Spider Mites",
        "symptoms": [
            "Leaves par fine yellow/white speckling appear ho sakti hai.",
            "Severe infestation me leaves bronze/dry ho sakti hain.",
            "Fine webbing visible ho sakti hai."
        ],
        "treatment": [
            "Leaf undersides carefully inspect karein.",
            "Affected plants ko isolate/monitor karein where practical.",
            "Plant stress aur excessive dryness minimize karein.",
            "Beneficial insects ko unnecessarily harm karne wale broad-spectrum pesticide use se bachein.",
            "Severe infestation me locally appropriate mite-management options agricultural expert se confirm karein."
        ],
        "prevention": [
            "Regular scouting karein.",
            "Leaf undersides inspect karein.",
            "Plant stress minimize karein.",
            "Beneficial organisms conserve karein."
        ]
    },

    "Tomato___Target_Spot": {
        "title": "🍅 Tomato Target Spot",
        "symptoms": [
            "Leaves par circular brown spots develop ho sakte hain.",
            "Lesions me target-like concentric pattern ho sakta hai.",
            "Fruit par dark sunken lesions develop ho sakte hain."
        ],
        "treatment": [
            "Affected leaves remove where practical.",
            "Plant debris remove karein.",
            "Good airflow maintain karein.",
            "Overhead watering avoid karein.",
            "Disease severe ho to locally approved fungicide options expert se confirm karein."
        ],
        "prevention": [
            "Crop rotation follow karein.",
            "Good spacing maintain karein.",
            "Field sanitation maintain karein.",
            "Regular scouting karein."
        ]
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "🍅 Tomato Yellow Leaf Curl Virus",
        "symptoms": [
            "Young leaves curl upward ho sakti hain.",
            "Leaves yellow ho sakti hain.",
            "Plant growth aur fruit production reduce ho sakti hai."
        ],
        "treatment": [
            "Virus ke liye direct curative treatment generally available nahi hota.",
            "Severely affected plants ko local agricultural guidance ke according remove/manage karein.",
            "Whitefly/vector population ko monitor karein.",
            "Nearby plants ko regularly inspect karein.",
            "Random antiviral ya pesticide claims par rely na karein."
        ],
        "prevention": [
            "Healthy planting material use karein.",
            "Vector monitoring aur management karein.",
            "Weeds that can host vectors ko manage karein.",
            "Early infected plants ko promptly identify karein."
        ]
    },

    "Tomato___Tomato_mosaic_virus": {
        "title": "🍅 Tomato Mosaic Virus",
        "symptoms": [
            "Leaves par light/dark green mosaic pattern appear ho sakta hai.",
            "Leaves distorted ya curled ho sakti hain.",
            "Plant growth aur fruit quality affect ho sakti hai."
        ],
        "treatment": [
            "Infected plants ko identify karke local agricultural guidance follow karein.",
            "Hands aur tools ko plants ke beech clean karein.",
            "Infected plant debris ko remove karein.",
            "Seeds/transplants ke source ko carefully manage karein.",
            "Virus ke liye random chemical treatment use na karein."
        ],
        "prevention": [
            "Certified disease-free seed/planting material use karein.",
            "Tools sanitize karein.",
            "Infected plant material ko spread na hone dein.",
            "Regular scouting karein."
        ]
    },

    "Tomato___healthy": {
        "title": "🍅 Tomato — Healthy Crop",
        "symptoms": [
            "Visible disease symptoms detect nahi hue."
        ],
        "treatment": [
            "Disease-specific treatment ki zarurat nahi hai.",
            "Normal irrigation, nutrition aur crop monitoring continue karein."
        ],
        "prevention": [
            "Regular leaf inspection karein.",
            "Good airflow maintain karein.",
            "Overwatering avoid karein.",
            "Healthy planting material use karein."
        ]
    }
}
predicted_disease = st.session_state.get("predicted_disease", None)

if predicted_disease and predicted_disease in disease_info:

    info = disease_info[predicted_disease]

    st.success(f"🌿 {info['title']}")

    st.markdown("### 🔍 Symptoms")
    for item in info["symptoms"]:
        st.write("•", item)

    st.markdown("### 🩺 Treatment / Management")
    for i, item in enumerate(info["treatment"], 1):
        st.write(f"**Step {i}:** {item}")

    st.markdown("### 🛡️ Prevention")
    for item in info["prevention"]:
        st.write("•", item)

    st.warning(
        "⚠️ **Important:** AI prediction ko final diagnosis na samjhein. "
        "Treatment, especially pesticide/fungicide/insecticide use se pehle "
        "ek baar local agriculture expert / plant pathologist se consult zaroor karein."
    )

else:

    st.info(
        "🔍 Pehle Disease Detection tab me leaf image upload karke "
        "analysis run karein. Uske baad yahan complete disease-management "
        "guidance show hogi."
    )
 
# ==================================================
# TAB 3 - FARMER ASSISTANT
# ==================================================
with tab3:

    st.subheader("🤖 AI Farmer Assistant")

    st.write(
        "Ask questions about crop diseases, symptoms and basic crop-care practices."
    )

    language = st.radio(
        "Choose language / भाषा चुनें",
        ["English", "हिंदी"],
        horizontal=True
    )

    if language == "English":

        question = st.text_input(
            "Ask your question",
            placeholder="Example: What should I do if my tomato leaves have spots?"
        )

        if question:

            st.info(
                "🤖 AI Assistant will be connected here in the next version."
            )

            st.write(
                "Your question:",
                question
            )

    else:

        question = st.text_input(
            "अपना सवाल पूछें",
            placeholder="उदाहरण: टमाटर के पत्तों पर दाग हैं, क्या करूँ?"
        )

        if question:

            st.info(
                "🤖 AI Assistant अगले version में Hindi में जवाब देगा।"
            )

            st.write(
                "आपका सवाल:",
                question
            )

    st.caption(
        "💡 Next upgrade: Hindi + English conversational AI assistant."
    )

# ==================================================
# TAB 4 - CONTACT EXPERT
# ==================================================
with tab4:

    st.subheader("👨‍🌾 Contact an Agriculture Expert")

    st.write(
        "AI prediction हमेशा final diagnosis नहीं होती. "
        "Serious crop damage की स्थिति में agriculture expert से सलाह लें."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🏢 Local Agriculture Support")

        st.write(
            "Contact your local Agriculture Department, "
            "Krishi Vigyan Kendra (KVK), agricultural university, "
            "or qualified crop specialist."
        )

    with col2:

        st.markdown("### 📋 Before Contacting an Expert")

        st.write(
            "Keep these details ready:"
        )

        st.write("• Crop name")
        st.write("• Location")
        st.write("• Disease symptoms")
        st.write("• Photos of affected leaves")
        st.write("• Approximate affected area")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "🌱 AI Crop Doctor | Crop Disease Detection using Deep Learning"
)
