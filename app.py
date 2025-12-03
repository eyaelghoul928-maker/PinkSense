import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="PinkSense | Analyse de Tumeur",
    layout="wide",
    page_icon="🎀"
)

# --- 1. CSS : THÈME PINKSENSE PREMIUM (rose gold + fuchsia + blanc) ---
st.markdown("""
<style>

/* --- GLOBAL --- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #FDF7FB; 
    color: #333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* --- TITRES --- */
h1 {
    color: #C2185B;
    border-bottom: 2px solid #FF80AB;
    padding-bottom: 8px;
    font-weight: 800;
}
h2, h3, h4 {
    color: #E91E63;
    font-weight: 600;
}

/* --- SIDEBAR : VERSION PREMIUM PINKSENSE --- */
[data-testid="stSidebar"] {
    background-color: white !important;
    padding: 20px;
    border-right: 3px solid #F8BBD0; 
    box-shadow: 4px 0 18px rgba(233, 30, 99, 0.12);
}

[data-testid="stSidebar"] .css-1d391kg, 
[data-testid="stSidebar"] .css-16idsys, 
[data-testid="stSidebar"] * {
    color: #C2185B !important;
    font-weight: 600 !important;
}

/* TITRES DE LA SIDEBAR */
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #AD1457 !important;
    font-weight: 800 !important;
}

/* SLIDERS */
.stSidebar .stSlider > div > div > div {
    background-color: #F8BBD0 !important; /* rail rose clair */
}

.stSidebar .stSlider > div > div > div > div {
    background-color: #E91E63 !important; /* curseur fuchsia */
}

/* --- METRICS (CARDS) --- */
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #E91E3F;
    box-shadow: 0 4px 15px rgba(233, 30, 99, 0.15);
}

/* --- BARRE DE PROGRESSION --- */
.stProgress > div > div > div > div {
    background-color: #FF4081 !important;
}

/* --- RADAR COLORS FIX --- */
svg g.trace.scatterpolar path {
    stroke-width: 3px !important;
}

/* --- ALERTES --- */
[data-testid="stAlert"] {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# --- 2. MODÈLE : CHARGEMENT ET ENTRAÎNEMENT ---
@st.cache_resource
def charger_modele():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model.fit(X_scaled, y)

    return model, scaler, data.feature_names, df.mean(), df.std()

model, scaler, feature_names, mean_values, std_values = charger_modele()

# --- 3. SIDEBAR : PARAMÈTRES PATIENT ---
st.sidebar.markdown("## 🎀 Données Morphologiques")
st.sidebar.markdown("Ajustez les paramètres clés pour analyser le risque de malignité.")
st.sidebar.markdown("---")

seuil = st.sidebar.slider(
    "Seuil de Classification (Sensibilité)",
    0.0, 1.0, 0.5, format="%.2f",
    help="Probabilité minimum pour classer comme Maligne."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Dimensions de la Tumeur")


def input_slider(label, key, help_text=""):
    avg = float(mean_values[key])
    std = float(std_values[key])
    min_val = max(0.0, avg - 4 * std)
    max_val = avg + 4 * std
    return st.sidebar.slider(label, min_val, max_val, avg, format="%.3f", help=help_text)


radius = input_slider("Rayon Moyen (μm)", 'mean radius')
texture = input_slider("Texture Moyenne", 'mean texture')
perimeter = input_slider("Périmètre Moyen", 'mean perimeter')
area = input_slider("Aire Moyenne (μm²)", 'mean area')
concavity = input_slider("Concavité Moyenne", 'mean concavity')

st.sidebar.markdown("---")
st.sidebar.caption("Modèle : Régression Logistique — Dataset : Wisconsin Breast Cancer")

# --- 4. PRÉDICTION ---
input_dict = mean_values.to_dict()
input_dict['mean radius'] = radius
input_dict['mean texture'] = texture
input_dict['mean perimeter'] = perimeter
input_dict['mean area'] = area
input_dict['mean concavity'] = concavity

input_df = pd.DataFrame([input_dict], columns=feature_names)
input_scaled = scaler.transform(input_df)

proba = model.predict_proba(input_scaled)
malin_prob = proba[0][0]
is_malignant = malin_prob > seuil

# --- 5. AFFICHAGE PRINCIPAL ---
st.title("PinkSense : Analyse IA du Risque de Tumeur 🎀")
st.markdown("Outil de démonstration IA pour illustrer l’analyse automatisée du risque de malignité.")

col_diag, col_radar = st.columns([1.5, 2.5])

# A. DIAGNOSTIC
with col_diag:
    st.subheader("🎯 Résultat de la Classification")

    if is_malignant:
        st.error("RISQUE MALIGNITÉ : POSITIF", icon="🚨")

        st.markdown(f"""
        <div class="stMetric" style="border-left-color: #D32F2F;">
            <p style="font-size: 0.9em; color: #555;">Probabilité de Malignité</p>
            <p style="font-size: 2.2em; font-weight: 700; color: #D32F2F;">{malin_prob*100:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(malin_prob * 100))

        if malin_prob >= 0.75:
            st.warning("Risque très élevé. Biopsie urgente recommandée.", icon="⚠️")
        else:
            st.info("Risque modéré. Surveillance nécessaire.", icon="🔍")

    else:
        st.success("RISQUE MALIGNITÉ : NÉGATIF", icon="✅")

        st.markdown(f"""
        <div class="stMetric" style="border-left-color: #388E3C;">
            <p style="font-size: 0.9em; color: #555;">Probabilité de Malignité</p>
            <p style="font-size: 2.2em; font-weight: 700; color: #388E3C;">{malin_prob*100:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(malin_prob * 100))
        st.info("Risque faible. Continuer le suivi clinique.", icon="👍")

# B. RADAR
with col_radar:
    st.subheader(" Profil Comparatif des Caractéristiques")
    st.caption("Visualisation des valeurs patient vs moyenne du dataset.")

    categories = ['Rayon', 'Texture', 'Périmètre', 'Aire', 'Concavité']
    keys = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity']
    patient_vals = [radius, texture, perimeter, area, concavity]
    avg_vals = [float(mean_values[k]) for k in keys]

    patient_rel = [p/a for p, a in zip(patient_vals, avg_vals)]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(categories),
        theta=categories,
        fill='toself',
        name='Moyenne Dataset',
        line_color='#00A3C9',
        opacity=0.3
    ))

    fig.add_trace(go.Scatterpolar(
        r=patient_rel,
        theta=categories,
        fill='toself',
        name='Cas Patient',
        line_color='#D32F2F' if is_malignant else '#388E3C',
        opacity=0.85
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        height=550,
        margin=dict(t=20, b=20, l=40, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 6. INTERPRÉTABILITÉ ---
st.subheader("🧠 Interprétabilité : Variables les Plus Influentes")

col_interpret_1, col_interpret_2 = st.columns([2, 1])

with col_interpret_1:
    coefs = model.coef_[0]
    importance_df = pd.DataFrame({'Variable': feature_names, 'Poids': coefs})
    importance_df['Abs_Poids'] = abs(importance_df['Poids'])
    importance_df = importance_df.sort_values(by='Abs_Poids', ascending=False).head(10)

    fig_bar = px.bar(
        importance_df.sort_values(by='Poids', ascending=False),
        x='Poids',
        y='Variable',
        orientation='h',
        color='Poids',
        color_continuous_scale=[(0, '#388E3C'), (0.5, 'white'), (1, '#D32F2F')],
        text_auto='.2f'
    )

    fig_bar.update_layout(height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_interpret_2:
    st.markdown("""
    Les poids indiquent l’influence de chaque variable :

    - **Poids positifs (rouges)** → favorisent la classification *Maligne*
    - **Poids négatifs (verts)** → favorisent la classification *Bénigne*

    Cela permet de comprendre l’impact morphologique réel.
    """)

st.markdown("---")

