"""
Advanced UI Components for PlannerIA - Modern, Professional Design
"""

import streamlit as st
from typing import Dict, Any, List

def inject_custom_css():
    """Inject modern, professional CSS styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling - Clean white background */
    .main {
        background: #ffffff;
        min-height: 100vh;
    }
    
    .stApp {
        background: #ffffff;
    }
    
    /* Dark text for readability */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #374151 !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Main title - High contrast, large, readable */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 4.5rem;
        background: linear-gradient(135deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 3rem 0 1rem 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.4rem;
        color: #4b5563;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 4rem;
        line-height: 1.5;
    }
    
    /* Professional cards - High contrast */
    .glass-card {
        background: #ffffff;
        border-radius: 16px;
        border: 2px solid #e5e7eb;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        border-color: #3b82f6;
    }
    
    /* Metrics styling - Large and readable */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-radius: 20px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
        border-color: #3b82f6;
    }
    
    .metric-value {
        font-size: 3.5rem !important;
        font-weight: 800;
        color: #1e293b !important;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 1.2rem !important;
        color: #64748b !important;
        margin-top: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status indicators - High contrast */
    .status-success {
        background: #10b981;
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    
    .status-error {
        background: #ef4444;
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
    }
    
    /* AI Agent cards - High contrast */
    .agent-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
        border-color: #3b82f6;
    }
    
    .agent-title {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    /* Chat styling - Clean and readable */
    .chat-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 3rem;
        margin: 3rem 0;
        border: 2px solid #e5e7eb;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    }
    
    /* Button enhancements - Large and readable */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 1rem 2.5rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25);
        transition: all 0.3s ease;
        min-height: 3rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35) !important;
        background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        box-shadow: 0 6px 24px rgba(245, 158, 11, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }
    
    /* Progress indicators */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        font-weight: 600;
        color: white;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.2));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global font size increases for readability */
    .stMarkdown {
        font-size: 1.2rem !important;
    }
    
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 1.2rem 2.5rem !important;
        min-height: 3.5rem !important;
    }
    
    .stSelectbox label, .stTextInput label, .stTextArea label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    .stMetric label {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    .stMetric .metric-value {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        font-size: 1.2rem !important;
    }
    
    /* Tab styling - larger text */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
    }
    
    /* Text styling - High contrast dark text */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.8rem !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #374151 !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .loading-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Modern card hover effects */
    .insight-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .insight-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
        border-color: rgba(255,255,255,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

def create_hero_section():
    """Create stunning hero section"""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0; background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 24px; margin: 2rem 0; border: 1px solid #cbd5e1;">
        <div class="main-title">🤖 PlannerIA</div>
        <div style="font-size: 2.5rem; font-weight: 600; color: #374151; margin-bottom: 1.5rem;">
            Advanced AI Suite
        </div>
        <div class="subtitle">
            🧠 Multi-Agent Intelligence • 📊 Professional Reporting • 🛡️ Enterprise-Grade Systems
        </div>
        <div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 1.5rem 3rem; border-radius: 50px; display: inline-block; margin-top: 2rem; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.25);">
            <span style="color: #10b981; font-weight: 700; font-size: 1.2rem;">●</span> 
            <span style="color: white; font-weight: 600; font-size: 1.2rem;">20 Systèmes IA Coordonnés</span> 
            <span style="margin: 0 1.5rem; color: rgba(255,255,255,0.5); font-size: 1.2rem;">|</span>
            <span style="color: #fbbf24; font-weight: 700; font-size: 1.2rem;">●</span> 
            <span style="color: white; font-weight: 600; font-size: 1.2rem;">Intelligence Prédictive</span>
            <span style="margin: 0 1.5rem; color: rgba(255,255,255,0.5); font-size: 1.2rem;">|</span>
            <span style="color: #a855f7; font-weight: 700; font-size: 1.2rem;">●</span> 
            <span style="color: white; font-weight: 600; font-size: 1.2rem;">Exportation Professionnelle</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_stunning_metrics(overview: Dict[str, Any]):
    """Create visually stunning metrics display"""
    
    st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="text-align: center; color: #1e293b; font-weight: 600; margin-bottom: 2rem; font-size: 2.5rem;">
            📊 Aperçu du Projet
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duration = overview.get('total_duration', 0)
        # Formater la durée pour éviter trop de décimales
        if isinstance(duration, (int, float)):
            duration_formatted = f"{duration:.0f}"
        else:
            duration_formatted = str(duration)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">⏱️</div>
            <div class="metric-value">{duration_formatted}</div>
            <div class="metric-label">Jours</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cost = overview.get('total_cost', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">💰</div>
            <div class="metric-value">€{cost:,.0f}</div>
            <div class="metric-label">Budget Total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        critical_path = overview.get('critical_path_duration', 0)
        # Formater le chemin critique pour éviter trop de décimales
        if isinstance(critical_path, (int, float)):
            critical_path_formatted = f"{critical_path:.0f}"
        else:
            critical_path_formatted = str(critical_path)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
            <div class="metric-value">{critical_path_formatted}</div>
            <div class="metric-label">Chemin Critique</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        phases = overview.get('phase_count', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📋</div>
            <div class="metric-value">{phases}</div>
            <div class="metric-label">Phases</div>
        </div>
        """, unsafe_allow_html=True)

def create_ai_agent_showcase(agents_data: Dict[str, Any]):
    """Create stunning AI agent showcase"""
    
    st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="text-align: center; color: #1e293b; font-weight: 600; margin-bottom: 2rem; font-size: 2.5rem;">
            🧠 Intelligence Multi-Agents
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Core agents
    core_agents = [
        ("🎯", "Supervisor", "Coordination intelligente", "#6366f1"),
        ("📋", "Planner", "Architecture WBS", "#8b5cf6"),
        ("💰", "Estimator", "Prédictions ML", "#10b981"),
        ("⚠️", "Risk", "Analyse prédictive", "#ef4444"),
        ("📝", "Documentation", "Génération automatique", "#f59e0b")
    ]
    
    advanced_agents = [
        ("🎯", "Strategy Advisor", "Positionnement marché", "#06b6d4"),
        ("🔄", "Learning Agent", "Amélioration continue", "#84cc16"),
        ("🌐", "Stakeholder Intel", "Analyse complexité", "#ec4899"),
        ("⚡", "Real-time Monitor", "Surveillance KPI", "#f97316"),
        ("🎨", "Innovation Catalyst", "Opportunités émergentes", "#a855f7")
    ]
    
    # Core agents display - High contrast
    st.markdown("""
    <div style="background: #ffffff; padding: 3rem; border-radius: 20px; margin: 3rem 0; border: 2px solid #e5e7eb; box-shadow: 0 10px 25px rgba(0,0,0,0.08);">
        <h3 style="color: #1e293b; text-align: center; margin-bottom: 3rem; font-size: 2rem; font-weight: 700;">🤖 Agents Core</h3>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i, (emoji, name, desc, color) in enumerate(core_agents):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 16px; margin: 0.5rem; border: 1px solid #e2e8f0; transition: all 0.3s ease;">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">{emoji}</div>
                <div style="color: {color}; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.5rem;">{name}</div>
                <div style="color: #64748b; font-size: 1rem; font-weight: 500;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Advanced agents display - High contrast  
    st.markdown("""
    <div style="background: #ffffff; padding: 3rem; border-radius: 20px; margin: 2rem 0; border: 2px solid #e5e7eb; box-shadow: 0 10px 25px rgba(0,0,0,0.08);">
        <h3 style="color: #1e293b; text-align: center; margin-bottom: 3rem; font-size: 2rem; font-weight: 700;">🚀 Agents Avancés</h3>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i, (emoji, name, desc, color) in enumerate(advanced_agents):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 16px; margin: 0.5rem; border: 1px solid #e2e8f0; transition: all 0.3s ease;">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">{emoji}</div>
                <div style="color: {color}; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.5rem;">{name}</div>
                <div style="color: #64748b; font-size: 1rem; font-weight: 500;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_export_section():
    """Create stunning export interface"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.1)); 
                padding: 2rem; border-radius: 20px; margin: 2rem 0; 
                backdrop-filter: blur(15px); border: 1px solid rgba(16,185,129,0.2);">
        <h3 style="color: white; text-align: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
            📄 Exports Professionnels
        </h3>
        <p style="text-align: center; color: rgba(255,255,255,0.8); margin-bottom: 2rem;">
            Générez des rapports de niveau entreprise pour vos parties prenantes
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_stunning_chat_interface():
    """Create beautiful chat interface"""
    st.markdown("""
    <div class="chat-container">
        <h3 style="color: #1e293b; text-align: center; margin-bottom: 2rem; font-size: 2.2rem; font-weight: 700;">
            💬 Intelligence Conversationnelle
        </h3>
        <p style="text-align: center; color: #64748b; font-size: 1.3rem; font-weight: 500;">
            Décrivez votre vision, laissez l'IA créer le plan parfait
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_demo_scenarios():
    """Create stunning demo scenarios section"""
    st.markdown("""
    <div style="background: #ffffff; padding: 3rem; border-radius: 20px; margin: 3rem 0; 
                border: 2px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.08);">
        <h3 style="color: #1e293b; text-align: center; margin-bottom: 2rem; font-size: 2rem; font-weight: 700;">🎬 Scénarios de Démonstration Avancés</h3>
        <p style="text-align: center; color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;">
            🏆 Projets complexes pour démonstration des capacités IA
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo buttons - Row 1: Classical scenarios
    st.markdown("### 🌟 Scénarios Classiques")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        if st.button("🏥 MedAI", help="Système médical IA", key="demo_medai"):
            demo_text = """MedAI : Système intelligent de support clinique multimodal

Concept :
Entrées multimodales : texte (notes cliniques), PDF (publications médicales), images (IRM, radios).
Pipeline IA : OCR + embeddings + classification multimodale (CLIP, Vision Transformers).

Agents spécialisés :
- Analyse de symptômes (NLP)
- Recherche documentaire (RAG sur PubMed)  
- Prédiction diagnostique (modèle ML supervisé)
- Génération de rapport médical structuré

Conformité : explication XAI (SHAP/LIME) pour justifier les résultats."""
            st.session_state['demo_input'] = demo_text
            st.info("✅ Interface mise à jour")
    
    with demo_col2:
        if st.button("💰 FinTech", help="Application financière", key="demo_fintech"):
            demo_text = """Application FinTech pour inclusion financière dans les marchés émergents

Objectifs :
- Paiements mobiles sans friction
- Microcrédits automatisés par IA
- Éducation financière personnalisée
- Conformité réglementaire multi-pays

Technologies :
- Blockchain pour sécurité
- ML pour scoring crédit
- NLP pour chatbot multilingue
- API bancaires ouvertes

Budget cible : €50k, équipe 5 développeurs"""
            st.session_state['demo_input'] = demo_text
            st.info("✅ Interface mise à jour")
    
    with demo_col3:
        if st.button("🚀 Startup", help="Plateforme SaaS", key="demo_startup"):
            demo_text = """Plateforme SaaS de gestion intelligente pour PME

Vision :
Automatiser la gestion administrative des petites entreprises avec IA

Modules clés :
- Comptabilité automatisée (OCR + ML)
- CRM avec prédiction de vente
- RH avec matching automatique
- Tableaux de bord prédictifs

Modèle : Freemium, 10€/mois/utilisateur
Cible : 100k utilisateurs en 18 mois"""
            st.session_state['demo_input'] = demo_text
            st.info("✅ Interface mise à jour")
    
    # Demo buttons - Row 2: Innovative scenarios
    st.markdown("### 🔥 Scénarios Innovants & Créatifs")
    demo_col4, demo_col5, demo_col6 = st.columns(3)
    
    with demo_col4:
        if st.button("🌱 EcoAI", help="IA environnementale révolutionnaire", key="demo_ecoai"):
            demo_text = """EcoAI : Plateforme d'optimisation carbone urbaine par IA satellitaire

Vision Révolutionnaire :
Réduction 40% émissions CO2 des villes via IA prédictive temps réel

Technologies Disruptives :
- Imagerie satellite haute résolution (analyse quotidienne)
- Computer Vision pour détection pollution/déforestation 
- ML predictif : modèles météo + trafic + consommation
- Jumeaux numériques urbains en 3D temps réel
- Blockchain pour crédits carbone automatisés

Applications Concrètes :
- Prédiction pics pollution 72h à l'avance
- Optimisation routes livraisons (-30% CO2)
- Détection feux forêts 15min avant signalement humain
- Recommandations personnalisées citoyens (app mobile)

Business Model : B2G (villes) + B2B (entreprises ESG)
Impact : 50 villes pilotes, 10M habitants, €200M économisés"""
            st.session_state['demo_input'] = demo_text
            st.info("✅ Interface mise à jour")
    
    with demo_col5:
        if st.button("🧠 NeuroTech", help="Interface cerveau-machine nouvelle génération", key="demo_neurotech"):
            demo_text = """NeuroTech : Interface cerveau-machine pour handicaps moteurs avec IA adaptative

Innovation Breakthrough :
Premier système BCI non-invasif avec apprentissage temps réel des patterns neurologiques

Architecture Technique Avancée :
- EEG haute densité 256 électrodes + ML temps réel
- Algorithmes d'adaptation automatique aux signaux cérébraux
- Prédiction intention motrice 200ms avant mouvement conscient
- Interface haptique bidirectionnelle (retour sensoriel)
- IA conversationnelle multimodale (pensée → parole → action)

Applications Révolutionnaires :
- Contrôle fauteuil roulant par pensée pure (précision 98%)
- Synthèse vocale depuis signaux neurologiques
- Manipulation robotique fine (prendre un verre d'eau)
- Réalité virtuelle thérapeutique personnalisée
- Communication directe cerveau-to-cerveau (expérimental)

Pipeline Clinique :
Phase 1 : Tests 50 patients hôpital Pitié-Salpêtrière
Phase 2 : Homologation CE medical (18 mois)
Phase 3 : Production 10k unités/an

Équipe : 8 neuroscientifiques + 12 ingénieurs IA
Budget : €2.5M sur 3 ans, financement EIC Pathfinder"""
            st.session_state['demo_input'] = demo_text
            st.info("✅ Interface mise à jour")
    
    with demo_col6:
        if st.button("🚗 QuantumMobility", help="Transport autonome quantique", key="demo_quantum"):
            demo_text = """QuantumMobility : Réseau de transport autonome optimisé par calcul quantique

Concept Futuriste :
Premier écosystème transport utilisant algorithmes quantiques pour optimisation trafic global

Révolution Quantique :
- Algorithme QAOA (Quantum Approximate Optimization) pour routage
- Résolution 10000x plus rapide problèmes NP-difficiles
- Optimisation simultanée 1M+ véhicules autonomes
- Prédiction quantique des patterns de mobilité urbaine
- Intrication quantique pour communication véhicules instantanée

Technologies Hybrides :
- Processeurs quantiques IBM/Google via cloud
- Edge computing classique pour décisions temps réel
- Capteurs LiDAR 4D + caméras neuromorphiques
- 5G/6G pour coordination flottes masives
- Digital twins quantiques des infrastructures

Applications Disruptives :
- Zéro embouteillage mathématiquement garanti
- Réduction 90% accidents (coordination parfaite)
- Optimisation énergétique multi-modale (train/bus/auto)
- Prédiction demande transport 7 jours à l'avance
- Tarification dynamique temps réel équitable

Déploiement :
Phase Alpha : Singapour (ville test, 2025)
Phase Beta : 5 smart cities européennes (2026-2027)  
Phase Production : 50 métropoles mondiales (2028)

Partenaires : Volkswagen, Uber, IBM Quantum Network
Financement : €50M (Série A), valorisation €500M"""
            st.session_state['demo_input'] = demo_text
            st.info("✅ Interface mise à jour")

def create_footer():
    """Create professional footer"""
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem; padding: 3rem; 
                background: #f8fafc; border-radius: 20px; 
                border: 2px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.05);">
        <div style="color: #1e293b; font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem;">
            🎓 PlannerIA Advanced AI Suite
        </div>
        <div style="color: #64748b; font-size: 1.3rem; font-weight: 500;">
            Projet Bootcamp IA Générative 2025 • Intelligence Artificielle Multi-Agents
        </div>
        <div style="margin-top: 2rem; padding: 1.5rem; background: #ffffff; 
                    border-radius: 16px; display: inline-block; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <span style="color: #10b981; font-weight: 700; font-size: 1.2rem;">🚀 Système Opérationnel</span> • 
            <span style="color: #f59e0b; font-weight: 700; font-size: 1.2rem;">20 Systèmes IA</span> • 
            <span style="color: #8b5cf6; font-weight: 700; font-size: 1.2rem;">Niveau Enterprise</span>
        </div>
    </div>
    """, unsafe_allow_html=True)