"""
Advanced Speech Recognition System for PlannerIA
Reconnaissance vocale intelligente avec support multilingue
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import queue

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    
    try:
        import pyaudio
        PYAUDIO_AVAILABLE = True
    except ImportError:
        PYAUDIO_AVAILABLE = False
        
    SPEECH_AVAILABLE = SPEECH_RECOGNITION_AVAILABLE  # Can work without pyaudio
except ImportError:
    SPEECH_AVAILABLE = False
    SPEECH_RECOGNITION_AVAILABLE = False
    PYAUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class RecognitionState(Enum):
    IDLE = "idle"
    LISTENING = "listening" 
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class Language(Enum):
    FRENCH = "fr-FR"
    ENGLISH = "en-US"
    AUTO = "auto"

@dataclass
class VoiceCommand:
    text: str
    confidence: float
    language: str
    timestamp: datetime
    duration: float
    audio_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

@dataclass 
class RecognitionResult:
    success: bool
    command: Optional[VoiceCommand] = None
    error: Optional[str] = None
    alternatives: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'success': self.success,
            'error': self.error,
            'alternatives': self.alternatives or []
        }
        if self.command:
            result['command'] = self.command.to_dict()
        return result

class VoiceSpeechRecognizer:
    """Système de reconnaissance vocale avancé avec support temps réel"""
    
    def __init__(self, language: Language = Language.FRENCH):
        if not SPEECH_AVAILABLE:
            raise ImportError("Module speech_recognition requis")
            
        self.language = language
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.recognition_thread: Optional[threading.Thread] = None
        self.command_queue = queue.Queue()
        self.callbacks: List[Callable] = []
        self.state = RecognitionState.IDLE
        self.pyaudio_available = PYAUDIO_AVAILABLE
        
        # Configuration optimisée
        self.recognizer.energy_threshold = 4000  # Seuil bruit ambiant
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Pause entre mots
        self.recognizer.phrase_threshold = 0.3  # Début phrase
        self.recognizer.non_speaking_duration = 0.5  # Silence fin
        
        # Statistiques
        self.stats = {
            'total_commands': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'average_confidence': 0.0,
            'last_recognition': None,
            'pyaudio_available': self.pyaudio_available
        }
        
        self._initialize_microphone()
        
    def _initialize_microphone(self):
        """Initialiser et calibrer le microphone"""
        if not self.pyaudio_available:
            logger.warning("⚠️ PyAudio non disponible - microphone désactivé")
            return
            
        try:
            self.microphone = sr.Microphone()
            
            # Calibration automatique du bruit ambiant
            logger.info("🎤 Calibration microphone...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            logger.info(f"✅ Microphone initialisé - Seuil: {self.recognizer.energy_threshold}")
            
        except Exception as e:
            logger.warning(f"Microphone non disponible: {e}")
            self.microphone = None
            
    def add_command_callback(self, callback: Callable):
        """Ajouter un callback pour les commandes reconnues"""
        self.callbacks.append(callback)
        
    def start_continuous_recognition(self):
        """Démarrer l'écoute continue"""
        if self.is_listening:
            logger.warning("Reconnaissance déjà active")
            return
            
        self.is_listening = True
        self.state = RecognitionState.LISTENING
        self.recognition_thread = threading.Thread(
            target=self._continuous_recognition_loop,
            daemon=True
        )
        self.recognition_thread.start()
        logger.info("🔍 Reconnaissance vocale continue activée")
        
    def stop_continuous_recognition(self):
        """Arrêter l'écoute continue"""
        self.is_listening = False
        self.state = RecognitionState.IDLE
        
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2)
            
        logger.info("⏹️ Reconnaissance vocale arrêtée")
        
    def _continuous_recognition_loop(self):
        """Boucle principale de reconnaissance continue"""
        while self.is_listening:
            try:
                # Utiliser un timeout plus court pour éviter les blocages
                result = self.recognize_single_command(timeout=2)
                
                if result.success and result.command:
                    # Ajouter à la queue
                    self.command_queue.put(result.command)
                    
                    # Notifier les callbacks
                    for callback in self.callbacks:
                        try:
                            callback(result.command)
                        except Exception as e:
                            logger.error(f"Erreur callback: {e}")
                            
                    # Mettre à jour statistiques
                    self._update_stats(result.command, success=True)
                    
                time.sleep(0.2)  # Pause plus longue entre les écoutes
                
            except Exception as e:
                logger.error(f"Erreur boucle reconnaissance: {e}")
                self._update_stats(None, success=False)
                time.sleep(1.0)  # Pause plus longue en cas d'erreur
                
    def recognize_single_command(self, timeout: float = 5.0) -> RecognitionResult:
        """Reconnaissance d'une seule commande"""
        if not self.microphone:
            return RecognitionResult(
                success=False,
                error="Microphone non disponible"
            )
            
        start_time = time.time()
        self.state = RecognitionState.LISTENING
        
        try:
            logger.debug("🎤 Écoute commande vocale...")
            
            # Créer une nouvelle instance de microphone pour éviter les conflits de contexte
            mic = sr.Microphone()
            with mic as source:
                # Ajustement rapide du bruit ambiant
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Écouter l'audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=10
                )
                
            duration = time.time() - start_time
            self.state = RecognitionState.PROCESSING
            
            # Reconnaissance avec Google (fallback vers autres services)
            text = self._recognize_with_fallback(audio)
            
            if not text:
                return RecognitionResult(
                    success=False,
                    error="Aucun texte reconnu"
                )
                
            # Créer commande
            command = VoiceCommand(
                text=text.strip(),
                confidence=0.8,  # Estimation
                language=self.language.value,
                timestamp=datetime.now(),
                duration=duration,
                audio_level=self.recognizer.energy_threshold
            )
            
            self.state = RecognitionState.COMPLETED
            logger.info(f"✅ Commande reconnue: '{command.text}'")
            
            return RecognitionResult(success=True, command=command)
            
        except sr.WaitTimeoutError:
            return RecognitionResult(
                success=False,
                error="Timeout - aucune parole détectée"
            )
        except sr.UnknownValueError:
            return RecognitionResult(
                success=False,
                error="Parole non comprise"
            )
        except sr.RequestError as e:
            return RecognitionResult(
                success=False,
                error=f"Erreur service reconnaissance: {e}"
            )
        except Exception as e:
            logger.error(f"Erreur reconnaissance: {e}")
            return RecognitionResult(
                success=False,
                error=f"Erreur inattendue: {e}"
            )
        finally:
            self.state = RecognitionState.IDLE
            
    def _recognize_with_fallback(self, audio) -> Optional[str]:
        """Reconnaissance avec fallback sur plusieurs services"""
        
        # Essayer Google (gratuit, limité)
        try:
            text = self.recognizer.recognize_google(
                audio, 
                language=self.language.value if self.language != Language.AUTO else "fr-FR"
            )
            if text:
                return text
        except:
            pass
            
        # Fallback: reconnaissance locale (moins précise)
        try:
            # Note: nécessiterait pocketsphinx pour fonctionner offline
            # text = self.recognizer.recognize_sphinx(audio, language="fr-FR")
            # return text
            pass
        except:
            pass
            
        return None
        
    def get_recent_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtenir les commandes récentes"""
        commands = []
        temp_queue = queue.Queue()
        
        # Vider la queue en sauvegardant
        while not self.command_queue.empty() and len(commands) < limit:
            try:
                cmd = self.command_queue.get_nowait()
                commands.append(cmd.to_dict())
                temp_queue.put(cmd)
            except queue.Empty:
                break
                
        # Remettre dans la queue
        while not temp_queue.empty():
            self.command_queue.put(temp_queue.get_nowait())
            
        return commands[-limit:]
        
    def _update_stats(self, command: Optional[VoiceCommand], success: bool):
        """Mettre à jour les statistiques"""
        self.stats['total_commands'] += 1
        
        if success and command:
            self.stats['successful_recognitions'] += 1
            self.stats['last_recognition'] = datetime.now().isoformat()
            
            # Moyenne glissante de confiance
            old_avg = self.stats['average_confidence']
            count = self.stats['successful_recognitions']
            self.stats['average_confidence'] = (
                (old_avg * (count - 1) + command.confidence) / count
            )
        else:
            self.stats['failed_recognitions'] += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de reconnaissance"""
        total = self.stats['total_commands']
        success_rate = (
            self.stats['successful_recognitions'] / total 
            if total > 0 else 0
        )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'current_state': self.state.value,
            'is_listening': self.is_listening,
            'queue_size': self.command_queue.qsize()
        }
        
    def test_microphone(self) -> Dict[str, Any]:
        """Tester la configuration du microphone"""
        if not self.pyaudio_available:
            return {
                'success': False,
                'error': 'PyAudio non disponible'
            }
            
        try:
            # Créer une nouvelle instance pour le test
            test_mic = sr.Microphone()
            with test_mic as source:
                # Test niveau audio
                logger.info("🔍 Test microphone - parlez maintenant...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                
                return {
                    'success': True,
                    'energy_threshold': self.recognizer.energy_threshold,
                    'audio_length': len(audio.get_raw_data()),
                    'sample_rate': audio.sample_rate,
                    'sample_width': audio.sample_width
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Instance globale
global_voice_recognizer: Optional[VoiceSpeechRecognizer] = None

def get_voice_recognizer(language: Language = Language.FRENCH) -> VoiceSpeechRecognizer:
    """Obtenir l'instance globale du recognizer"""
    global global_voice_recognizer
    
    if not SPEECH_AVAILABLE:
        raise ImportError("Modules speech_recognition et pyaudio non disponibles")
        
    if global_voice_recognizer is None:
        global_voice_recognizer = VoiceSpeechRecognizer(language)
        
    return global_voice_recognizer

def start_voice_recognition():
    """Démarrer la reconnaissance globale"""
    recognizer = get_voice_recognizer()
    recognizer.start_continuous_recognition()

def stop_voice_recognition():
    """Arrêter la reconnaissance globale"""
    if global_voice_recognizer:
        global_voice_recognizer.stop_continuous_recognition()