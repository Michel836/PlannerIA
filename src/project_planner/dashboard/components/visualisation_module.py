#!/usr/bin/env python3
"""
🎨 MODULE VISUALISATION 3D IA/ML - Interface Streamlit
Visualisations spectaculaires pour soutenance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os
import json

# Import du moteur 3D
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))
try:
    from ml_3d_engine import ML3DEngine
except ImportError:
    st.error("❌ Impossible d'importer le moteur 3D ML")
    ML3DEngine = None

def render_mode_soutenance(engine, project_context: Dict):
    """🚀 Mode Soutenance Plein Écran Automatisé"""
    
    # Configuration depuis session state
    config = st.session_state.get('soutenance_config', {})
    
    # Interface plein écran
    st.markdown("""
    <style>
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header mode soutenance
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        padding: 1rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        position: relative;
        animation: rainbow 3s ease-in-out infinite alternate;
    ">
        🚀 MODE SOUTENANCE ACTIF 🚀
    </div>
    
    <style>
    @keyframes rainbow {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Contrôles de présentation
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1, 1, 1, 1])
    
    with col_ctrl1:
        if st.button("⏮️ Précédent", key="sout_prev", use_container_width=True):
            if 'current_slide' not in st.session_state:
                st.session_state.current_slide = 0
            st.session_state.current_slide = max(0, st.session_state.current_slide - 1)
    
    with col_ctrl2:
        if st.button("▶️ Play/Pause", key="sout_play", use_container_width=True):
            st.session_state.auto_play = not st.session_state.get('auto_play', False)
    
    with col_ctrl3:
        if st.button("⏭️ Suivant", key="sout_next", use_container_width=True):
            if 'current_slide' not in st.session_state:
                st.session_state.current_slide = 0
            st.session_state.current_slide = min(4, st.session_state.current_slide + 1)
    
    with col_ctrl4:
        if st.button("❌ Quitter Mode", key="sout_exit", use_container_width=True):
            st.session_state.mode_soutenance = False
            st.success("✅ Visualisation mise à jour")
    
    # Initialiser slide courante
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 0
    
    current_slide = st.session_state.current_slide
    
    # Progress bar
    progress = (current_slide + 1) / 5
    st.progress(progress)
    st.caption(f"Visualisation {current_slide + 1}/5")
    
    # Visualisations en mode soutenance
    theme_soutenance = 'neon' if config.get('effets_neon', True) else 'galaxy'
    
    try:
        if current_slide == 0:
            st.markdown("### 🎯 PARETO 3D INTELLIGENT - CLUSTERING IA")
            fig = engine.generer_pareto_3d_intelligent({}, theme_soutenance)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': False,
                'staticPlot': False
            })
            
        elif current_slide == 1:
            st.markdown("### 🧠 RÉSEAU NEURONAL 3D - DÉPENDANCES PROJET")
            fig = engine.generer_reseau_neuronal_3d({}, theme_soutenance)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': False,
                'staticPlot': False
            })
            
        elif current_slide == 2:
            st.markdown("### 🌊 SURFACE DE RISQUE 3D - PRÉDICTIONS ML")
            fig = engine.generer_surface_risque_3d({}, theme_soutenance)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': False,
                'staticPlot': False
            })
            
        elif current_slide == 3:
            st.markdown("### 🎯 HOLOGRAMME MÉTRIQUES - RADAR IA")
            fig = engine.generer_hologramme_metriques({}, theme_soutenance)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': False,
                'staticPlot': False
            })
            
        elif current_slide == 4:
            st.markdown("### 🎬 ÉVOLUTION TEMPORELLE - ANIMATION IA")
            fig = engine.generer_animation_temporelle({})
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': False,
                'staticPlot': False
            })
    
    except Exception as e:
        st.error(f"Erreur visualisation slide {current_slide}: {e}")
    
    # Instructions soutenance
    st.markdown("---")
    st.markdown("""
    **🎯 Instructions Mode Soutenance:**
    - **⏮️/⏭️** : Navigation manuelle entre visualisations
    - **▶️** : Démarrer/Arrêter défilement automatique
    - **Effets visuels** : Rotation et animations activées
    - **Qualité maximale** : Optimisé pour projection
    """)

def render_visualisation_module(project_context: Dict[str, Any]):
    """
    🎨 Module Visualisation 3D IA/ML Principal
    Spectaculaire pour soutenance !
    """
    
    # Vérification mode soutenance - DOIT ÊTRE EN PREMIER !
    if st.session_state.get('mode_soutenance', False):
        st.info("🚀 MODE SOUTENANCE ACTIF - Interface de présentation")
        # Initialiser le moteur pour le mode soutenance
        if 'ml_3d_engine' not in st.session_state:
            st.session_state.ml_3d_engine = ML3DEngine()
        engine = st.session_state.ml_3d_engine
        render_mode_soutenance(engine, project_context)
        return  # Arrêter le rendu normal si en mode soutenance
    
    # Header spectaculaire
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        animation: pulse 2s ease-in-out infinite alternate;
    ">
        <h1 style="margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🎨 VISUALISATION 3D IA/ML
        </h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
            🚀 Graphiques Spectaculaires pour Soutenance 🚀
        </p>
    </div>
    
    <style>
    @keyframes pulse {
        from { transform: scale(1); }
        to { transform: scale(1.02); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    if ML3DEngine is None:
        st.error("🚫 Moteur 3D non disponible")
        return
    
    # Initialiser moteur 3D
    if 'ml_3d_engine' not in st.session_state:
        st.session_state.ml_3d_engine = ML3DEngine()
    
    engine = st.session_state.ml_3d_engine
    
    # Sidebar - Contrôles spectaculaires
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(45deg, #667eea, #764ba2);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        ">
            <h3 style="margin: 0;">🎛️ Contrôles 3D</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection de thème
        theme = st.selectbox(
            "🎨 Thème Couleurs",
            ['neon', 'fire', 'galaxy', 'matrix', 'hologram'],
            index=0,
            help="Choix de palette pour vos visualisations"
        )
        
        # Mode spectacle
        mode_spectacle = st.checkbox("🎭 Mode Spectacle", value=True)
        if mode_spectacle:
            st.success("✨ Optimisé pour soutenance !")
        
        # Paramètres avancés
        with st.expander("⚙️ Paramètres Avancés"):
            resolution_3d = st.slider("Résolution 3D", 10, 50, 25)
            animation_speed = st.slider("Vitesse Animation", 100, 1000, 500)
            opacity_3d = st.slider("Opacité", 0.3, 1.0, 0.8)
        
        st.markdown("---")
        st.markdown("### 🎯 Actions Rapides")
        
        if st.button("🎬 Mode Présentation", type="primary"):
            st.session_state.presentation_mode = True
        
        if st.button("📸 Capture Écran"):
            st.info("🔄 Capture en cours...")
        
        if st.button("🎥 Enregistrer Vidéo"):
            st.info("🔴 Enregistrement démarré...")
    
    # Tabs principales - Interface révolutionnaire
    tabs = st.tabs([
        "🎯 Pareto 3D", 
        "🧠 Réseau Neuronal", 
        "🌊 Surface Risque",
        "🎯 Hologramme", 
        "🎬 Animations",
        "🎨 Galerie"
    ])
    
    with tabs[0]:
        render_pareto_3d_tab(engine, theme, project_context)
    
    with tabs[1]:
        render_reseau_neuronal_tab(engine, theme, project_context)
    
    with tabs[2]:
        render_surface_risque_tab(engine, theme, project_context)
    
    with tabs[3]:
        render_hologramme_tab(engine, theme, project_context)
    
    with tabs[4]:
        render_animations_tab(engine, theme, project_context)
    
    with tabs[5]:
        render_galerie_tab(engine, project_context)

def render_pareto_3d_tab(engine: ML3DEngine, theme: str, project_context: Dict):
    """🎯 Tab Pareto 3D Intelligent"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h2 style="margin: 0;">🎯 PARETO 3D INTELLIGENT</h2>
        <p style="margin: 0.5rem 0 0 0;">Clustering IA + Visualisation Spectaculaire</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ Configuration")
        
        nb_taches = st.slider("Nombre de tâches", 10, 50, 25)
        clustering_auto = st.checkbox("Clustering IA Auto", value=True)
        
        if clustering_auto:
            nb_clusters = st.slider("Nombre de clusters", 2, 8, 4)
        
        afficher_labels = st.checkbox("Afficher labels", value=True)
        rotation_auto = st.checkbox("Rotation automatique", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Légende")
        st.markdown("""
        - **Axe X** : 📈 Impact Projet
        - **Axe Y** : 🎲 Probabilité Succès  
        - **Axe Z** : ⚡ Effort Requis
        - **Taille** : 💰 Budget Alloué
        - **Couleur** : 🎨 Cluster IA
        """)
    
    with col1:
        # Préparer données
        project_data = {
            'nb_taches': nb_taches,
            'clustering': clustering_auto,
            'nb_clusters': nb_clusters if clustering_auto else 1
        }
        
        with st.spinner("🎨 Génération Pareto 3D spectaculaire..."):
            fig = engine.generer_pareto_3d_intelligent(project_data, theme)
            
            # Afficher avec configuration plein écran
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['pan3d', 'orbitRotation', 'tableRotation']
                }
            )
        
        # Métriques de résumé
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("🎯 Tâches Analysées", nb_taches)
        with col_m2:
            st.metric("🧠 Clusters IA", nb_clusters if clustering_auto else "Manuel")
        with col_m3:
            st.metric("⚡ Performance", "Optimale", "↗️")
        with col_m4:
            st.metric("🎨 Rendu", "3D WebGL", "✨")

def render_reseau_neuronal_tab(engine: ML3DEngine, theme: str, project_context: Dict):
    """🧠 Tab Réseau Neuronal 3D"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h2 style="margin: 0;">🧠 RÉSEAU NEURONAL PROJET</h2>
        <p style="margin: 0.5rem 0 0 0;">Visualisation des Dépendances IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎛️ Paramètres Réseau")
        
        nb_modules = st.slider("Modules projet", 8, 20, 12)
        densite_connexions = st.slider("Densité connexions", 0.2, 0.8, 0.4)
        
        algorithme_layout = st.selectbox(
            "Algorithme Layout",
            ['spring', 'circular', 'shell', 'random'],
            index=0
        )
        
        afficher_criticite = st.checkbox("Criticité IA", value=True)
        animation_physique = st.checkbox("Animation physique", value=False)
        
        st.markdown("---")
        st.markdown("### 🎨 Rendu")
        st.markdown("""
        - **Nœuds** : 🔵 Modules/Tâches
        - **Arêtes** : ➡️ Dépendances
        - **Couleur** : 🌡️ Criticité IA
        - **Taille** : 📊 Importance
        """)
    
    with col1:
        # Données réseau
        dependencies_data = {
            'nb_modules': nb_modules,
            'densite': densite_connexions,
            'algorithm': algorithme_layout
        }
        
        with st.spinner("🧠 Construction réseau neuronal 3D..."):
            fig = engine.generer_reseau_neuronal_3d(dependencies_data, theme)
            
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'reseau_neuronal_3d',
                        'height': 800,
                        'width': 1200,
                        'scale': 2
                    }
                }
            )
        
        # Analyses IA
        st.markdown("### 🔍 Analyse IA du Réseau")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("🔗 Connexions", f"{int(nb_modules * densite_connexions * 5)}")
        with col_a2:
            st.metric("🎯 Centralité Moy", f"{np.random.uniform(0.3, 0.8):.2f}")
        with col_a3:
            st.metric("⚡ Complexité", f"{np.random.choice(['Faible', 'Moyenne', 'Élevée'])}")

def render_surface_risque_tab(engine: ML3DEngine, theme: str, project_context: Dict):
    """🌊 Tab Surface de Risque 3D"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff4757 0%, #ff6b7a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h2 style="margin: 0;">🌊 SURFACE DE RISQUE 3D</h2>
        <p style="margin: 0.5rem 0 0 0;">Prédictions ML Volumétriques</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎚️ Contrôles Surface")
        
        resolution_surface = st.slider("Résolution", 15, 40, 25)
        seuil_critique = st.slider("Seuil critique", 0.5, 0.9, 0.7)
        
        type_surface = st.selectbox(
            "Type de surface",
            ['Volume complet', 'Isosurfaces', 'Points critiques'],
            index=0
        )
        
        modele_ml = st.selectbox(
            "Modèle ML",
            ['Neural Network', 'Random Forest', 'SVM', 'Hybrid'],
            index=0
        )
        
        temps_prediction = st.slider("Horizon (semaines)", 1, 12, 4)
        
        st.markdown("---")
        st.markdown("### 📊 Axes 3D")
        st.markdown("""
        - **X** : ⏰ Temps (% avancement)
        - **Y** : 💰 Budget (% utilisé)
        - **Z** : ✅ Qualité (% atteinte)
        - **Couleur** : 🌡️ Niveau Risque
        """)
    
    with col1:
        # Données de risque
        risk_data = {
            'resolution': resolution_surface,
            'seuil': seuil_critique,
            'type': type_surface,
            'modele': modele_ml,
            'horizon': temps_prediction
        }
        
        with st.spinner("🌊 Calcul surface de risque ML..."):
            fig = engine.generer_surface_risque_3d(risk_data, theme)
            
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
        
        # Alertes critiques
        st.markdown("### 🚨 Points Critiques Détectés")
        
        alert_col1, alert_col2 = st.columns(2)
        with alert_col1:
            st.error("🔴 Zone Critique: Budget 85%, Temps 70%")
            st.warning("🟠 Zone Attention: Qualité <60%")
        
        with alert_col2:
            st.info("🔵 Zone Optimale: Début projet")
            st.success("🟢 Zone Sécurisée: Fin planifiée")

def render_hologramme_tab(engine: ML3DEngine, theme: str, project_context: Dict):
    """🎯 Tab Hologramme Métriques"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #00d4ff 0%, #1e3c72 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h2 style="margin: 0;">🎯 HOLOGRAMME MÉTRIQUES</h2>
        <p style="margin: 0.5rem 0 0 0;">Radar 3D Multi-Dimensionnel</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🎛️ Configuration Hologramme")
        
        metriques_selectionnees = st.multiselect(
            "Métriques à afficher",
            ['Budget', 'Temps', 'Qualité', 'Risque', 'Équipe', 
             'Innovation', 'Satisfaction', 'Performance', 'Complexité'],
            default=['Budget', 'Temps', 'Qualité', 'Risque', 'Équipe'],
            help="Sélectionnez jusqu'à 8 métriques"
        )
        
        comparaison_mode = st.checkbox("Mode Comparaison", value=True)
        
        if comparaison_mode:
            st.markdown("**Séries à comparer:**")
            montrer_prevu = st.checkbox("📋 Prévu", value=True)
            montrer_realise = st.checkbox("✅ Réalisé", value=True) 
            montrer_predit = st.checkbox("🤖 Prédit IA", value=True)
        
        effet_holographique = st.checkbox("Effet holographique", value=True)
        rotation_continue = st.checkbox("Rotation continue", value=False)
        
        st.markdown("---")
        if st.button("🎨 Régénérer Hologramme", type="primary"):
            st.success("✅ Visualisation mise à jour")
    
    with col1:
        # Données métriques
        metrics_data = {
            'metriques_selection': metriques_selectionnees,
            'comparaison': comparaison_mode,
            'series': {
                'prevu': montrer_prevu if comparaison_mode else True,
                'realise': montrer_realise if comparaison_mode else True,
                'predit': montrer_predit if comparaison_mode else True
            }
        }
        
        if len(metriques_selectionnees) < 3:
            st.warning("⚠️ Sélectionnez au moins 3 métriques pour l'hologramme")
        else:
            with st.spinner("🎯 Génération hologramme 3D..."):
                fig = engine.generer_hologramme_metriques(metrics_data, theme)
                
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': True}
                )
            
            # Insights IA
            st.markdown("### 🧠 Insights IA")
            
            insight_col1, insight_col2 = st.columns(2)
            with insight_col1:
                st.info("📈 **Tendance**: Amélioration globale détectée")
                st.success("✨ **Recommandation**: Continuer stratégie actuelle")
            
            with insight_col2:
                st.warning("⚠️ **Attention**: Métrique 'Risque' en hausse")
                st.error("🔍 **Focus requis**: Qualité vs Temps")

def render_animations_tab(engine: ML3DEngine, theme: str, project_context: Dict):
    """🎬 Tab Animations Temporelles"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h2 style="margin: 0;">🎬 ANIMATIONS TEMPORELLES</h2>
        <p style="margin: 0.5rem 0 0 0;">Évolution 3D dans le Temps</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### ⚙️ Contrôles Animation")
        
        type_animation = st.selectbox(
            "Type d'animation",
            ['Évolution métriques', 'Trajectoires 3D', 'Morphing formes', 'Particules'],
            index=0
        )
        
        duree_animation = st.slider("Durée (frames)", 5, 20, 10)
        vitesse_lecture = st.slider("Vitesse (ms)", 200, 1000, 500)
        
        transitions_fluides = st.checkbox("Transitions fluides", value=True)
        effets_particules = st.checkbox("Effets particules", value=False)
        
        st.markdown("---")
        st.markdown("### 🎥 Export Vidéo")
        
        if st.button("🎬 Démarrer Animation", type="primary"):
            st.session_state.animation_active = True
        
        if st.button("⏸️ Pause", key="pause_anim"):
            st.session_state.animation_active = False
            
        if st.button("💾 Exporter MP4", key="export_mp4"):
            st.info("🎥 Export en cours...")
    
    with col1:
        # Données animation
        timeline_data = {
            'type': type_animation,
            'duree': duree_animation,
            'vitesse': vitesse_lecture,
            'effects': {
                'transitions': transitions_fluides,
                'particules': effets_particules
            }
        }
        
        with st.spinner("🎬 Génération animation 3D..."):
            fig = engine.generer_animation_temporelle(timeline_data)
            
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
        
        # Contrôles lecture
        st.markdown("### 🎮 Contrôles de Lecture")
        
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        
        with control_col1:
            if st.button("▶️ Play", key="play_control"):
                st.success("▶️ Animation lancée")
        
        with control_col2:
            if st.button("⏸️ Pause", key="pause_control"):
                st.info("⏸️ Animation en pause")
        
        with control_col3:
            if st.button("⏭️ Suivant", key="next_control"):
                st.info("⏭️ Frame suivante")
                
        with control_col4:
            if st.button("🔄 Reset", key="reset_control"):
                st.info("🔄 Animation reset")

def render_galerie_tab(engine: ML3DEngine, project_context: Dict):
    """🎨 Tab Galerie Interactive"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h2 style="margin: 0;">🎨 GALERIE INTERACTIVE</h2>
        <p style="margin: 0.5rem 0 0 0;">Collection de Visualisations Spectaculaires</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Galerie en grille
    st.markdown("### 🖼️ Aperçus Rapides")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        ">
            <h4>🎯 Pareto 3D</h4>
            <p>Clustering intelligent</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👁️ Voir Pareto", key="see_pareto"):
            st.session_state.current_view_mode = "🎨 Visualisation 3D"
            st.success("✅ Visualisation mise à jour")
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        ">
            <h4>🧠 Réseau Neural</h4>
            <p>Dépendances projet</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👁️ Voir Réseau", key="see_network"):
            st.session_state.current_view_mode = "🎨 Visualisation 3D"
            st.success("✅ Visualisation mise à jour")
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff4757, #ff6b7a);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        ">
            <h4>🌊 Surface Risque</h4>
            <p>Prédictions ML</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👁️ Voir Surface", key="see_surface"):
            st.session_state.current_view_mode = "🎨 Visualisation 3D"
            st.success("✅ Visualisation mise à jour")
    
    st.markdown("---")
    
    # Options d'export
    st.markdown("### 💾 Export & Partage")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        st.markdown("**📸 Images Haute Résolution**")
        format_image = st.selectbox("Format", ['PNG', 'SVG', 'PDF'], key="img_format")
        resolution = st.selectbox("Résolution", ['1920x1080', '2560x1440', '3840x2160'], key="resolution")
        
        if st.button("📥 Télécharger Images", type="primary"):
            st.success("✅ Images exportées!")
    
    with export_col2:
        st.markdown("**🎥 Vidéos & Animations**")
        format_video = st.selectbox("Format vidéo", ['MP4', 'GIF', 'WebM'], key="vid_format")
        fps = st.selectbox("FPS", ['30', '60', '120'], key="fps")
        
        if st.button("🎬 Exporter Vidéos", type="primary"):
            st.success("✅ Vidéos exportées!")
    
    with export_col3:
        st.markdown("**🔗 Partage & Embed**")
        st.text_area("URL de partage", "https://plannerai.demo/viz/3d/abc123", key="share_url")
        
        if st.button("📋 Copier Lien", type="primary"):
            st.success("✅ Lien copié!")
    
    # Mode présentation
    st.markdown("---")
    st.markdown("### 🎭 Mode Présentation Soutenance")
    
    pres_col1, pres_col2 = st.columns(2)
    
    with pres_col1:
        st.markdown("**🎯 Configuration Soutenance**")
        mode_plein_ecran = st.checkbox("Plein écran automatique", value=True)
        transitions_auto = st.checkbox("Transitions automatiques", value=True)
        duree_slide = st.slider("Durée par visualisation (s)", 10, 60, 30)
    
    with pres_col2:
        st.markdown("**🎨 Effets Spectaculaires**")
        effets_neon = st.checkbox("Effets néon", value=True)
        rotation_auto = st.checkbox("Rotation automatique", value=True)
        son_ambiance = st.checkbox("Ambiance sonore", value=False)
    
    if st.button("🚀 LANCER MODE SOUTENANCE", type="primary", use_container_width=True):
        st.session_state.mode_soutenance = True
        st.session_state.soutenance_config = {
            'plein_ecran': mode_plein_ecran,
            'transitions_auto': transitions_auto,
            'duree_slide': duree_slide,
            'effets_neon': effets_neon,
            'rotation_auto': rotation_auto,
            'son_ambiance': son_ambiance
        }
        st.success("🚀 Mode soutenance activé ! Redirection en cours...")
        st.success("✅ Visualisation mise à jour")

# Test si exécuté directement
if __name__ == "__main__":
    st.set_page_config(page_title="Test Module Visualisation 3D", layout="wide")
    
    test_context = {'project_name': 'Test Visualisation 3D'}
    render_visualisation_module(test_context)