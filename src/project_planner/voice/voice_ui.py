"""
Advanced Voice UI Components for PlannerIA
Interface vocale intelligente pour Streamlit
"""

import streamlit as st
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading

try:
    from .text_to_speech import (
        get_voice_synthesizer,
        SpeechConfig,
        VoiceType,
        SpeechRate
    )
    from .voice_processor import (
        get_voice_processor,
        CommandIntent,
        ProcessingResult
    )
    
    # Try to import speech recognizer, but make it optional
    try:
        from .speech_recognizer import (
            get_voice_recognizer, 
            VoiceCommand, 
            RecognitionState, 
            Language
        )
        SPEECH_RECOGNITION_AVAILABLE = True
    except ImportError:
        SPEECH_RECOGNITION_AVAILABLE = False
        
    VOICE_AVAILABLE = True  # TTS and processing are available
except ImportError as e:
    VOICE_AVAILABLE = False
    SPEECH_RECOGNITION_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class VoiceUIManager:
    """Gestionnaire de l'interface vocale pour Streamlit"""
    
    def __init__(self):
        self.is_initialized = False
        self.last_command_time = None
        self.voice_session_active = False
        
        # Callbacks pour actions
        self.action_callbacks = {
            CommandIntent.GENERATE_PLAN: None,
            CommandIntent.EXPORT_PDF: None,
            CommandIntent.EXPORT_CSV: None,
            CommandIntent.START_MONITORING: None,
            CommandIntent.STOP_MONITORING: None,
            CommandIntent.SHOW_ANALYTICS: None,
            CommandIntent.SHOW_DASHBOARD: None,
            CommandIntent.READ_RESULTS: None
        }
        
    def initialize_voice_system(self):
        """Initialiser le système vocal"""
        if not VOICE_AVAILABLE:
            st.error("❌ Modules vocaux non disponibles")
            return False
            
        try:
            # Initialiser les composants de base (TTS et processing)
            if 'voice_synthesizer' not in st.session_state:
                config = SpeechConfig(
                    voice_type=VoiceType.FEMALE,
                    rate=SpeechRate.NORMAL.value,
                    volume=0.8
                )
                st.session_state.voice_synthesizer = get_voice_synthesizer(config)
                
            if 'voice_processor' not in st.session_state:
                st.session_state.voice_processor = get_voice_processor()
                self._setup_voice_handlers()
            
            # Reconnaissance vocale optionnelle
            if SPEECH_RECOGNITION_AVAILABLE:
                if 'voice_recognizer' not in st.session_state:
                    st.session_state.voice_recognizer = get_voice_recognizer(Language.FRENCH)
            else:
                st.warning("⚠️ Reconnaissance vocale indisponible (pyaudio manquant)")
                
            self.is_initialized = True
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur initialisation système vocal: {e}")
            logger.error(f"Erreur initialisation voice system: {e}")
            return False
            
    def _setup_voice_handlers(self):
        """Configurer les gestionnaires de commandes vocales"""
        processor = st.session_state.voice_processor
        
        # Handler pour génération de plan
        def handle_generate_plan(params, original_text):
            if 'project_description' in params:
                st.session_state.voice_command_result = {
                    'action': 'generate_plan',
                    'description': params['project_description'],
                    'timestamp': datetime.now().isoformat()
                }
            return {'action': 'plan_generation_started', 'response': 'Génération du plan démarrée'}
            
        # Handler pour export PDF
        def handle_export_pdf(params, original_text):
            st.session_state.voice_command_result = {
                'action': 'export_pdf',
                'timestamp': datetime.now().isoformat()
            }
            return {'action': 'pdf_export_started', 'response': 'Export PDF démarré'}
            
        # Handler pour surveillance
        def handle_start_monitoring(params, original_text):
            st.session_state.voice_command_result = {
                'action': 'start_monitoring',
                'timestamp': datetime.now().isoformat()
            }
            return {'action': 'monitoring_started', 'response': 'Surveillance démarrée'}
            
        # Handler pour lecture résultats
        def handle_read_results(params, original_text):
            if 'last_analysis' in st.session_state:
                synthesizer = st.session_state.voice_synthesizer
                synthesizer.speak_analysis_result(st.session_state.last_analysis, priority=2)
            return {'action': 'results_read', 'response': 'Lecture des résultats en cours'}
            
        # Enregistrer les handlers
        processor.register_action_handler(CommandIntent.GENERATE_PLAN, handle_generate_plan)
        processor.register_action_handler(CommandIntent.EXPORT_PDF, handle_export_pdf)
        processor.register_action_handler(CommandIntent.START_MONITORING, handle_start_monitoring)
        processor.register_action_handler(CommandIntent.READ_RESULTS, handle_read_results)
        
    def render_voice_control_panel(self):
        """Afficher le panneau de contrôle vocal"""
        if not self.initialize_voice_system():
            return
            
        st.markdown("---")
        st.markdown("### 🎤 Interface Vocale Intelligente")
        
        # Section principale : RECONNAISSANCE VOCALE (STT)
        st.markdown("#### 🎤 Reconnaissance Vocale (STT) - Parlez à PlannerIA")
        
        stt_col1, stt_col2 = st.columns([3, 2])
        
        with stt_col1:
            # Le bouton principal pour parler
            if st.button("🎤 PARLER À PLANNERAI", key="main_voice_btn", help="Cliquez et parlez maintenant!", type="primary"):
                self._handle_voice_listen()
                
            # Instructions claires
            st.info("💡 Cliquez le bouton et dites par exemple : 'Je souhaite créer une application mobile'")
            
        with stt_col2:
            # État reconnaissance
            if 'voice_recognizer' in st.session_state:
                recognizer = st.session_state.voice_recognizer
                if recognizer.is_listening:
                    st.error("🔴 **ÉCOUTE EN COURS** - Parlez maintenant!")
                else:
                    st.success("✅ **PRÊT À ÉCOUTER**")
            
        st.markdown("---")
        
        # Section secondaire : SYNTHÈSE VOCALE (TTS) - Tests et contrôles
        with st.expander("🔊 Contrôles Synthèse Vocale (TTS) - PlannerIA vous parle", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                st.markdown("**🎯 Tests TTS**")
            
                # Boutons de test TTS
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    # Bouton test vocal long
                    if st.button("🔊 Test Long", key="voice_test_btn"):
                        self._handle_voice_test()
                        
                with btn_col2:
                    # Bouton test rapide
                    if st.button("⚡ Test Rapide", key="voice_test_short_btn", help="Test TTS court"):
                        self._handle_voice_test_short()
                
            with col2:
                st.markdown("**🔧 Contrôles TTS**")
                
                control_col1, control_col2 = st.columns(2)
                
                with control_col1:
                    # Bouton stop vocal
                    if st.button("⏹️ Stop TTS", key="voice_stop_btn", help="Arrêter la synthèse vocale"):
                        self._handle_voice_stop()
                        st.rerun()
                        
                with control_col2:
                    # Bouton vider queue
                    if st.button("🗑️ Vider Queue", key="voice_clear_btn", help="Vider la file d'attente"):
                        self._handle_voice_clear()
                        st.rerun()
                        
            with col3:
                st.markdown("**📊 État TTS**")
                
                # État TTS en temps réel
                if 'voice_synthesizer' in st.session_state:
                    synthesizer = st.session_state.voice_synthesizer
                    if synthesizer.is_speaking:
                        st.error("🔊 **TTS ACTIF**")
                    elif synthesizer.speech_queue.qsize() > 0:
                        st.warning(f"📝 Queue: {synthesizer.speech_queue.qsize()}")
                    else:
                        st.success("🔇 **TTS LIBRE**")
                        
                # Bouton état détaillé
                if st.button("📊 État Détaillé", key="voice_status_btn", help="Diagnostic complet"):
                    self._show_voice_status()
            
            # Configuration vocale dans l'expander
            st.markdown("**⚙️ Configuration Voix**")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                voice_type = st.selectbox(
                    "Type de voix",
                    ["female", "male", "auto"],
                    key="voice_type_select"
                )
                
            with config_col2:
                speech_rate = st.slider(
                    "Vitesse (mots/min)",
                    120, 250, 180,
                    key="voice_rate_slider"
                )
            
        # Zone de statut vocal
        self._render_voice_status()
        
        # Historique des commandes
        self._render_command_history()
        
        # Aide contextuelle spécifique STT vs TTS
        self._render_voice_help_stt_tts()
        
    def _handle_voice_listen(self):
        """Gérer l'écoute d'une commande vocale"""
        if not self.is_initialized:
            st.error("Système vocal non initialisé")
            return
            
        if not SPEECH_RECOGNITION_AVAILABLE:
            st.error("❌ Reconnaissance vocale non disponible")
            return
            
        try:
            with st.spinner("🎤 Écoute en cours... Parlez maintenant"):
                recognizer = st.session_state.voice_recognizer
                result = recognizer.recognize_single_command(timeout=5.0)
                
            if result.success and result.command:
                # Traiter la commande
                processor = st.session_state.voice_processor
                processing_result = processor.process_command(result.command.text)
                
                # Afficher le résultat
                if processing_result.success:
                    st.success(f"✅ Commande comprise: '{result.command.text}'")
                    st.info(f"🎯 Action: {processing_result.intent.value}")
                    
                    # Feedback vocal
                    synthesizer = st.session_state.voice_synthesizer
                    synthesizer.speak_text(
                        processing_result.response_text or "Commande exécutée",
                        priority=1
                    )
                    
                else:
                    st.warning(f"⚠️ Commande non comprise: '{result.command.text}'")
                    st.error(f"Erreur: {processing_result.error}")
                    
            else:
                st.error(f"❌ Reconnaissance échouée: {result.error}")
                
        except Exception as e:
            st.error(f"❌ Erreur écoute vocale: {e}")
            logger.error(f"Erreur voice listen: {e}")
            
    def _handle_voice_test(self):
        """Tester la synthèse vocale"""
        try:
            synthesizer = st.session_state.voice_synthesizer
            
            # Test avec message plus long pour pouvoir tester le stop
            test_text = """Test de synthèse vocale PlannerIA. 
            Le système est maintenant parfaitement opérationnel avec reconnaissance vocale, 
            synthèse de texte, traitement des commandes naturelles en français, 
            et contrôles avancés incluant arrêt et vidage de queue. 
            Vous pouvez maintenant interagir vocalement avec l'intelligence artificielle."""
            
            synthesizer.speak_text(test_text, priority=2)
            st.success("🔊 Test vocal long lancé - Utilisez Stop pour l'interrompre")
            
        except Exception as e:
            st.error(f"❌ Erreur test vocal: {e}")
            
    def _handle_voice_test_short(self):
        """Test TTS rapide"""
        try:
            synthesizer = st.session_state.voice_synthesizer
            test_text = "Test rapide TTS PlannerIA."
            synthesizer.speak_text(test_text, priority=2)
            st.success("⚡ Test rapide lancé")
            
        except Exception as e:
            st.error(f"❌ Erreur test rapide: {e}")
            
    def _show_voice_status(self):
        """Afficher l'état détaillé du système vocal"""
        try:
            if 'voice_synthesizer' not in st.session_state:
                st.warning("⚠️ Synthétiseur non initialisé")
                return
                
            synthesizer = st.session_state.voice_synthesizer
            stats = synthesizer.get_stats()
            
            with st.expander("📊 État Détaillé TTS", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🔊 Actif", synthesizer.is_speaking)
                    st.metric("📝 Queue", synthesizer.speech_queue.qsize())
                    
                with col2:
                    st.metric("✅ Succès", stats.get('successful_speeches', 0))
                    st.metric("❌ Échecs", stats.get('failed_speeches', 0))
                    
                with col3:
                    st.metric("📊 Total", stats.get('total_speeches', 0))
                    st.metric("🎤 Voix", stats.get('available_voices', 0))
                
                # Afficher la dernière synthèse
                if 'last_speech' in stats and stats['last_speech']:
                    last = stats['last_speech']
                    st.text(f"Dernière: {last.get('text', '')[:50]}...")
                    
        except Exception as e:
            st.error(f"❌ Erreur affichage statut: {e}")
            
    def _handle_voice_stop(self):
        """Arrêter la synthèse vocale en cours"""
        try:
            if 'voice_synthesizer' not in st.session_state:
                st.warning("⚠️ Synthétiseur non initialisé")
                return
                
            synthesizer = st.session_state.voice_synthesizer
            was_speaking = synthesizer.is_speaking
            queue_size = synthesizer.speech_queue.qsize()
            
            # Arrêter la synthèse
            synthesizer.stop_current_speech()
            
            if was_speaking or queue_size > 0:
                st.success(f"⏹️ Synthèse arrêtée (était actif: {was_speaking}, queue: {queue_size})")
            else:
                st.info("⏹️ Aucune synthèse en cours à arrêter")
                
        except Exception as e:
            st.error(f"❌ Erreur arrêt vocal: {e}")
            logger.error(f"Erreur stop voice: {e}")
            
    def _handle_voice_clear(self):
        """Vider la queue de synthèse"""
        try:
            if 'voice_synthesizer' not in st.session_state:
                st.warning("⚠️ Synthétiseur non initialisé")
                return
                
            synthesizer = st.session_state.voice_synthesizer
            queue_size_before = synthesizer.speech_queue.qsize()
            
            # Vider la queue
            synthesizer.clear_queue()
            
            if queue_size_before > 0:
                st.success(f"🗑️ Queue vidée ({queue_size_before} éléments supprimés)")
            else:
                st.info("🗑️ Queue déjà vide")
                
        except Exception as e:
            st.error(f"❌ Erreur vidage queue: {e}")
            logger.error(f"Erreur clear voice: {e}")
            
    def _render_voice_stats(self):
        """Afficher les statistiques vocales"""
        try:
            if 'voice_recognizer' in st.session_state:
                recognizer_stats = st.session_state.voice_recognizer.get_stats()
                st.metric("Commandes", recognizer_stats['total_commands'])
                st.metric("Succès", f"{recognizer_stats['success_rate']:.0%}")
                
            if 'voice_synthesizer' in st.session_state:
                synthesizer_stats = st.session_state.voice_synthesizer.get_stats()
                st.metric("Synthèses", synthesizer_stats['total_speeches'])
                
        except Exception as e:
            logger.error(f"Erreur stats vocales: {e}")
            
    def _render_voice_status(self):
        """Afficher le statut du système vocal"""
        st.markdown("#### 🔍 Statut Système Vocal")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            if self.is_initialized:
                st.success("✅ Système vocal opérationnel")
            else:
                st.error("❌ Système vocal non initialisé")
                
        with status_col2:
            # Indicateur temps réel
            if 'voice_recognizer' in st.session_state:
                recognizer = st.session_state.voice_recognizer
                if recognizer.is_listening:
                    st.info("🔴 Écoute active")
                else:
                    st.info("⚪ En attente")
                    
            # Indicateur TTS
            if 'voice_synthesizer' in st.session_state:
                synthesizer = st.session_state.voice_synthesizer
                if synthesizer.is_speaking:
                    st.info("🔊 TTS actif")
                elif synthesizer.speech_queue.qsize() > 0:
                    st.info(f"📝 Queue: {synthesizer.speech_queue.qsize()}")
                else:
                    st.info("🔇 TTS libre")
                    
    def _render_command_history(self):
        """Afficher l'historique des commandes"""
        if 'voice_processor' not in st.session_state:
            return
            
        with st.expander("📜 Historique des Commandes Vocales", expanded=False):
            processor = st.session_state.voice_processor
            
            if hasattr(processor, 'context_history') and processor.context_history:
                # Afficher les 5 dernières commandes
                recent_commands = processor.context_history[-5:]
                
                for i, cmd in enumerate(reversed(recent_commands), 1):
                    success_icon = "✅" if cmd['success'] else "❌"
                    st.markdown(f"""
                    **{i}.** {success_icon} `{cmd['command']}`  
                    *Intent: {cmd['intent']} | Confiance: {cmd['confidence']:.2f}*
                    """)
            else:
                st.info("Aucune commande dans l'historique")
                
    def _render_voice_help_stt_tts(self):
        """Aide spécifique STT vs TTS"""
        with st.expander("❓ Comprendre STT vs TTS - Comment utiliser la voix", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎤 **STT (Speech-to-Text)**
                **= VOUS PARLEZ → PlannerIA comprend**
                
                **Comment faire :**
                1. Cliquez "🎤 PARLER À PLANNERAI"  
                2. Attendez le signal "🔴 ÉCOUTE EN COURS"
                3. **Parlez clairement** votre demande
                4. PlannerIA analyse et répond
                
                **Exemples de phrases :**
                - *"Je souhaite créer une application mobile"*
                - *"Génère un plan pour un site e-commerce"*
                - *"J'ai besoin d'aide pour un projet IA"*
                - *"Montre-moi les analyses"*
                """)
                
            with col2:
                st.markdown("""
                ### 🔊 **TTS (Text-to-Speech)**
                **= PlannerIA vous parle → VOUS entendez**
                
                **Utilisations :**
                - PlannerIA **lit** les résultats d'analyse
                - **Notifications** vocales d'alertes
                - **Tests** de fonctionnement vocal
                - **Feedback** audio des actions
                
                **Contrôles disponibles :**
                - ⚡ Test Rapide (court)
                - 🔊 Test Long (pour tester Stop)
                - ⏹️ Stop TTS (arrêter immédiatement)
                - 🗑️ Vider Queue (nettoyer)
                """)
                
            st.info("💡 **Résumé simple :** Cliquez 🎤 PARLER pour que PlannerIA vous **écoute** → Il vous **répond** en vocal (TTS) ET texte !")
            
        # Commandes détaillées dans un expander séparé
        with st.expander("📝 Liste Complète des Commandes Vocales", expanded=False):
            st.markdown("""
            **🎯 Génération de plans:**
            - "Je souhaite réaliser un projet [description]"
            - "Génère un plan pour [description]"
            - "J'ai besoin d'un plan pour application mobile"
            - "Comment faire une plateforme web"
            
            **📊 Navigation:**
            - "Montre les analyses" / "Affiche le dashboard"
            - "Va à la surveillance" / "Ouvre le monitoring"
            
            **📁 Actions:**
            - "Exporte en PDF" / "Exporte en CSV"
            - "Démarre la surveillance" / "Arrête le monitoring"
            
            **🔊 Informations:**
            - "Lis-moi les résultats" / "Quel est le statut"
            - "Aide" / "Comment utiliser"
            """)
            
    def render_voice_floating_button(self):
        """Bouton flottant pour activation rapide"""
        # CSS pour bouton flottant
        st.markdown("""
        <style>
        .voice-float-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            transition: all 0.3s ease;
        }
        .voice-float-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        }
        </style>
        """, unsafe_allow_html=True)
        
    def handle_voice_command_results(self):
        """Traiter les résultats de commandes vocales stockés en session"""
        if 'voice_command_result' not in st.session_state:
            return
            
        result = st.session_state.voice_command_result
        action = result.get('action')
        
        # Traiter selon le type d'action
        if action == 'generate_plan':
            description = result.get('description', '')
            if description:
                # Déclencher génération de plan via session state
                st.session_state.voice_plan_request = description
                st.rerun()
                
        elif action == 'export_pdf':
            # Déclencher export PDF
            st.session_state.voice_export_pdf = True
            st.rerun()
            
        elif action == 'start_monitoring':
            # Déclencher surveillance
            st.session_state.voice_start_monitoring = True
            st.rerun()
            
        # Nettoyer après traitement
        del st.session_state.voice_command_result

# Fonctions utilitaires pour intégration Streamlit
def render_voice_interface():
    """Fonction principale pour afficher l'interface vocale"""
    if not VOICE_AVAILABLE:
        st.warning("⚠️ Interface vocale non disponible - modules manquants")
        return
        
    if 'voice_ui_manager' not in st.session_state:
        st.session_state.voice_ui_manager = VoiceUIManager()
        
    manager = st.session_state.voice_ui_manager
    manager.render_voice_control_panel()
    manager.handle_voice_command_results()

def render_voice_floating_controls():
    """Afficher les contrôles vocaux flottants"""
    if not VOICE_AVAILABLE:
        return
        
    if 'voice_ui_manager' not in st.session_state:
        st.session_state.voice_ui_manager = VoiceUIManager()
        
    manager = st.session_state.voice_ui_manager
    manager.render_voice_floating_button()

def is_voice_available() -> bool:
    """Vérifier si le système vocal est disponible"""
    return VOICE_AVAILABLE

def get_voice_stats() -> Dict[str, Any]:
    """Obtenir les statistiques vocales consolidées"""
    if not VOICE_AVAILABLE:
        return {'available': False}
        
    stats = {'available': True}
    
    try:
        if 'voice_recognizer' in st.session_state:
            stats['recognition'] = st.session_state.voice_recognizer.get_stats()
            
        if 'voice_synthesizer' in st.session_state:
            stats['synthesis'] = st.session_state.voice_synthesizer.get_stats()
            
        if 'voice_processor' in st.session_state:
            stats['processing'] = st.session_state.voice_processor.get_stats()
            
    except Exception as e:
        logger.error(f"Erreur collecte stats vocales: {e}")
        stats['error'] = str(e)
        
    return stats