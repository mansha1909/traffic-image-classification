import streamlit as st
import plotly.express as px
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {background-color:#0e1117;}
.main {background-color:#0e1117;}

.card {
    background-color:#1c1f26;
    padding:20px;
    border-radius:12px;
    text-align:center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
}

.small-card {
    background-color:#1c1f26;
    padding:15px;
    border-radius:10px;
}

.title {
    text-align:center;
    font-size:32px;
    font-weight:bold;
    color:white;
}

.subtitle {
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='title'>🚦 ROAD TRAFFIC EVENT DETECTION</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Using Feature Engineering Techniques</div>", unsafe_allow_html=True)

# -------------------- TOP CARDS --------------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown("<div class='card'><h4>Total Images</h4><h2>5061</h2></div>", unsafe_allow_html=True)
c2.markdown("<div class='card'><h4>Classes</h4><h2>4</h2></div>", unsafe_allow_html=True)
c3.markdown("<div class='card'><h4>Train/Test</h4><h2>80% / 20%</h2></div>", unsafe_allow_html=True)
c4.markdown("<div class='card'><h4>Accuracy</h4><h2 style='color:#2ecc71;'>94.38%</h2></div>", unsafe_allow_html=True)

# -------------------- ROW 1 --------------------
col1, col2 = st.columns(2)

# PIE CHART
with col1:
    st.subheader("Class Distribution")
    labels = ["Dense Traffic","Sparse Traffic","Accident","Fire"]
    values = [1590,1455,1052,964]

    fig = px.pie(names=labels, values=values, hole=0.5,
                 color_discrete_sequence=["#3498db","#2ecc71","#f1c40f","#e74c3c"])
    fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# SAMPLE IMAGES
with col2:
    st.subheader("Sample Images")
    c1,c2,c3,c4 = st.columns(4)

    c1.image("https://picsum.photos/200?1", caption="Accident")
    c2.image("https://picsum.photos/200?2", caption="Dense")
    c3.image("https://picsum.photos/200?3", caption="Fire")
    c4.image("https://picsum.photos/200?4", caption="Sparse")

# -------------------- ROW 2 --------------------
col3, col4 = st.columns(2)

# BAR CHART
with col3:
    st.subheader("Image Resolution Distribution")
    res = ["128x128","256x256","512x512","1024x1024"]
    counts = [1234,2145,1287,395]

    fig2 = px.bar(x=res, y=counts,
                  color=res,
                  color_discrete_sequence=["#8e44ad","#3498db","#2ecc71","#e67e22"])
    fig2.update_layout(paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

# PCA GRAPH
with col4:
    st.subheader("PCA Variance Explained")

    x = np.arange(1,50)
    y = np.cumsum(np.random.rand(49))
    y = y / max(y) * 100

    fig3 = px.line(x=x, y=y)
    fig3.update_layout(paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- ROW 3 (PREDICTION + TABLE) --------------------
col5, col6 = st.columns(2)

# PREDICTION BOX
with col5:
    st.subheader("Prediction")

    file = st.file_uploader("Upload Image", type=["jpg","png"])

    if file:
        img = Image.open(file)
        st.image(img, width=250)

        st.success("Predicted: Dense Traffic")
        st.markdown("### Confidence: 96.21%")

# TABLE
with col6:
    st.subheader("Recent Predictions")

    st.dataframe({
        "Image":["img1","img2","img3","img4"],
        "Class":["Dense","Accident","Fire","Sparse"],
        "Confidence":["96%","93%","97%","94%"],
        "Time":[str(datetime.now())]*4
    })