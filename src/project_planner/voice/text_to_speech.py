"""
Advanced Text-to-Speech System for PlannerIA
Synthèse vocale intelligente avec personnalisation
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import json

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceType(Enum):
    FEMALE = "female"
    MALE = "male" 
    AUTO = "auto"

class SpeechRate(Enum):
    SLOW = 120
    NORMAL = 180
    FAST = 250

class Language(Enum):
    FRENCH = "french"
    ENGLISH = "english"

@dataclass
class SpeechConfig:
    voice_type: VoiceType = VoiceType.FEMALE
    rate: int = SpeechRate.NORMAL.value
    volume: float = 0.8
    language: Language = Language.FRENCH
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'voice_type': self.voice_type.value,
            'rate': self.rate,
            'volume': self.volume,
            'language': self.language.value
        }

@dataclass
class SpeechTask:
    text: str
    config: SpeechConfig
    priority: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class VoiceTextToSpeech:
    """Système de synthèse vocale avancé avec queue et personnalisation"""
    
    def __init__(self, config: SpeechConfig = None):
        if not TTS_AVAILABLE:
            raise ImportError("Module pyttsx3 requis pour TTS")
            
        self.config = config or SpeechConfig()
        self.engine = None
        self.speech_queue = queue.PriorityQueue()
        self.is_speaking = False
        self.speech_thread: Optional[threading.Thread] = None
        self.available_voices = {}
        self.task_counter = 0  # Compteur unique pour éviter les comparaisons d'objets
        
        # Statistiques
        self.stats = {
            'total_speeches': 0,
            'successful_speeches': 0,
            'failed_speeches': 0,
            'total_characters': 0,
            'last_speech': None
        }
        
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialiser le moteur TTS"""
        try:
            self.engine = pyttsx3.init()
            
            # Découvrir les voix disponibles
            voices = self.engine.getProperty('voices')
            
            for voice in voices:
                voice_id = voice.id
                voice_name = voice.name.lower()
                
                # Catégoriser par genre et langue
                if 'female' in voice_name or 'woman' in voice_name:
                    gender = VoiceType.FEMALE
                elif 'male' in voice_name or 'man' in voice_name:
                    gender = VoiceType.MALE
                else:
                    gender = VoiceType.AUTO
                    
                if 'french' in voice_name or 'fr' in voice_name:
                    language = Language.FRENCH
                else:
                    language = Language.ENGLISH
                    
                self.available_voices[voice_id] = {
                    'name': voice.name,
                    'gender': gender,
                    'language': language
                }
                
            logger.info(f"✅ TTS initialisé - {len(self.available_voices)} voix disponibles")
            self._apply_config(self.config)
            
        except Exception as e:
            logger.error(f"Erreur initialisation TTS: {e}")
            raise
            
    def _apply_config(self, config: SpeechConfig):
        """Appliquer la configuration au moteur"""
        if not self.engine:
            return
            
        try:
            # Sélectionner la meilleure voix
            best_voice = self._select_best_voice(config)
            if best_voice:
                self.engine.setProperty('voice', best_voice)
                
            # Configurer vitesse et volume
            self.engine.setProperty('rate', config.rate)
            self.engine.setProperty('volume', config.volume)
            
            logger.debug(f"Configuration TTS appliquée: {config.to_dict()}")
            
        except Exception as e:
            logger.error(f"Erreur application config TTS: {e}")
            
    def _select_best_voice(self, config: SpeechConfig) -> Optional[str]:
        """Sélectionner la meilleure voix selon la config"""
        
        # Filtrer par langue d'abord
        language_matches = [
            voice_id for voice_id, voice_info in self.available_voices.items()
            if voice_info['language'] == config.language
        ]
        
        if not language_matches:
            # Fallback: toutes les voix
            language_matches = list(self.available_voices.keys())
            
        if not language_matches:
            return None
            
        # Filtrer par genre
        if config.voice_type != VoiceType.AUTO:
            gender_matches = [
                voice_id for voice_id in language_matches
                if self.available_voices[voice_id]['gender'] == config.voice_type
            ]
            
            if gender_matches:
                return gender_matches[0]
                
        # Retourner le premier match de langue
        return language_matches[0]
        
    def speak_text(self, text: str, config: SpeechConfig = None, priority: int = 0):
        """Ajouter un texte à synthétiser dans la queue"""
        if not text.strip():
            logger.warning("Texte vide pour TTS")
            return
            
        task_config = config or self.config
        task = SpeechTask(
            text=text.strip(),
            config=task_config,
            priority=priority
        )
        
        # Ajouter à la queue avec compteur unique pour éviter comparaison d'objets
        self.task_counter += 1
        self.speech_queue.put((-priority, self.task_counter, task))
        
        # Démarrer le thread de synthèse si pas actif
        if not self.is_speaking:
            self._start_speech_thread()
            
        logger.debug(f"Ajouté à la queue TTS: '{text[:50]}...' (priorité: {priority})")
        
    def _start_speech_thread(self):
        """Démarrer le thread de synthèse"""
        if self.is_speaking and self.speech_thread and self.speech_thread.is_alive():
            return
            
        # Créer un nouveau thread à chaque fois (les threads ne sont pas réutilisables)
        self.speech_thread = threading.Thread(
            target=self._speech_loop,
            daemon=True
        )
        self.speech_thread.start()
        
    def _speech_loop(self):
        """Boucle principale de synthèse vocale"""
        self.is_speaking = True
        
        try:
            while not self.speech_queue.empty():
                try:
                    # Récupérer tâche suivante
                    _, _, task = self.speech_queue.get(timeout=1)
                    
                    # Appliquer configuration
                    self._apply_config(task.config)
                    
                    # Synthèse vocale
                    logger.info(f"🔊 Synthèse: '{task.text[:50]}...'")
                    start_time = time.time()
                    
                    self.engine.say(task.text)
                    self.engine.runAndWait()
                    
                    duration = time.time() - start_time
                    
                    # Mise à jour statistiques
                    self._update_stats(task, success=True, duration=duration)
                    
                    self.speech_queue.task_done()
                    
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Erreur synthèse vocale: {e}")
                    self._update_stats(task, success=False)
                    
        finally:
            self.is_speaking = False
            logger.debug("Thread synthèse vocale terminé")
            
    def speak_analysis_result(self, analysis_data: Dict[str, Any], priority: int = 1):
        """Synthétiser les résultats d'analyse IA de manière intelligente"""
        
        if not analysis_data:
            self.speak_text("Aucune donnée d'analyse disponible", priority=priority)
            return
            
        # Construire un résumé vocal intelligent
        summary_parts = []
        
        # Informations projet
        if 'project_name' in analysis_data:
            summary_parts.append(f"Analyse du projet {analysis_data['project_name']}")
            
        # Métriques clés
        if 'duration' in analysis_data:
            summary_parts.append(f"Durée estimée: {analysis_data['duration']} jours")
            
        if 'budget' in analysis_data:
            budget = analysis_data['budget']
            summary_parts.append(f"Budget: {budget:,.0f} euros")
            
        # Risques critiques
        if 'critical_risks' in analysis_data:
            risks = analysis_data['critical_risks']
            if risks:
                risk_count = len(risks)
                summary_parts.append(f"{risk_count} risque{'s' if risk_count > 1 else ''} critique{'s' if risk_count > 1 else ''} identifié{'s' if risk_count > 1 else ''}")
                
        # Recommandations
        if 'recommendations' in analysis_data:
            recommendations = analysis_data['recommendations']
            if recommendations:
                rec_count = len(recommendations[:3])  # Top 3
                summary_parts.append(f"{rec_count} recommandation{'s' if rec_count > 1 else ''} principale{'s' if rec_count > 1 else ''}")
                
        # Score de faisabilité
        if 'feasibility_score' in analysis_data:
            score = analysis_data['feasibility_score']
            if score >= 80:
                summary_parts.append("Faisabilité excellente")
            elif score >= 60:
                summary_parts.append("Faisabilité bonne")
            elif score >= 40:
                summary_parts.append("Faisabilité modérée")
            else:
                summary_parts.append("Faisabilité faible")
                
        if summary_parts:
            text = ". ".join(summary_parts) + "."
            self.speak_text(text, priority=priority)
        else:
            self.speak_text("Analyse complétée avec succès", priority=priority)
            
    def speak_alert(self, alert_data: Dict[str, Any], priority: int = 2):
        """Synthétiser une alerte système"""
        
        alert_level = alert_data.get('level', 'info')
        metric = alert_data.get('metric', 'système')
        message = alert_data.get('message', 'Alerte détectée')
        
        # Adapter le ton selon le niveau
        if alert_level == 'critical':
            intro = "Alerte critique!"
        elif alert_level == 'warning':
            intro = "Attention:"
        else:
            intro = "Information:"
            
        text = f"{intro} {message}"
        self.speak_text(text, priority=priority)
        
    def speak_command_feedback(self, command: str, result: str, priority: int = 0):
        """Donner un feedback vocal pour une commande"""
        
        feedback_templates = {
            'generate_plan': "Génération du plan en cours",
            'export_pdf': "Export PDF démarré", 
            'start_monitoring': "Surveillance activée",
            'stop_monitoring': "Surveillance arrêtée",
            'show_analytics': "Affichage des analyses",
            'error': "Une erreur s'est produite"
        }
        
        if result in feedback_templates:
            text = feedback_templates[result]
        else:
            text = f"Commande {command} exécutée"
            
        self.speak_text(text, priority=priority)
        
    def stop_current_speech(self):
        """Arrêter la synthèse en cours"""
        # Arrêter le flag pour interrompre la boucle
        self.is_speaking = False
        
        if self.engine:
            try:
                # Arrêter le moteur TTS
                self.engine.stop()
                logger.info("🔇 Synthèse vocale interrompue")
            except Exception as e:
                logger.error(f"Erreur arrêt TTS: {e}")
                
        # Attendre que le thread se termine proprement
        if self.speech_thread and self.speech_thread.is_alive():
            try:
                self.speech_thread.join(timeout=1)
            except:
                pass
                
    def clear_queue(self):
        """Vider la queue de synthèse"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
                
        logger.info("Queue TTS vidée")
        
    def _update_stats(self, task: SpeechTask, success: bool, duration: float = 0):
        """Mettre à jour les statistiques"""
        self.stats['total_speeches'] += 1
        self.stats['total_characters'] += len(task.text)
        
        if success:
            self.stats['successful_speeches'] += 1
            self.stats['last_speech'] = {
                'text': task.text[:100],
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
        else:
            self.stats['failed_speeches'] += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques TTS"""
        total = self.stats['total_speeches']
        success_rate = (
            self.stats['successful_speeches'] / total
            if total > 0 else 0
        )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'is_speaking': self.is_speaking,
            'queue_size': self.speech_queue.qsize(),
            'available_voices': len(self.available_voices)
        }
        
    def get_available_voices(self) -> Dict[str, Any]:
        """Obtenir la liste des voix disponibles"""
        return self.available_voices
        
    def test_voice(self, text: str = "Test de synthèse vocale PlannerIA") -> bool:
        """Tester la synthèse vocale"""
        try:
            self.speak_text(text, priority=10)  # Priorité haute pour test
            return True
        except Exception as e:
            logger.error(f"Erreur test TTS: {e}")
            return False

# Instance globale
global_voice_synthesizer: Optional[VoiceTextToSpeech] = None

def get_voice_synthesizer(config: SpeechConfig = None) -> VoiceTextToSpeech:
    """Obtenir l'instance globale du synthesizer"""
    global global_voice_synthesizer
    
    if not TTS_AVAILABLE:
        raise ImportError("Module pyttsx3 non disponible")
        
    if global_voice_synthesizer is None:
        global_voice_synthesizer = VoiceTextToSpeech(config)
        
    return global_voice_synthesizer

def speak_text(text: str, config: SpeechConfig = None, priority: int = 0):
    """Synthétiser un texte (fonction utilitaire)"""
    synthesizer = get_voice_synthesizer()
    synthesizer.speak_text(text, config, priority)

def speak_analysis_result(analysis_data: Dict[str, Any], priority: int = 1):
    """Synthétiser un résultat d'analyse (fonction utilitaire)"""
    synthesizer = get_voice_synthesizer()
    synthesizer.speak_analysis_result(analysis_data, priority)