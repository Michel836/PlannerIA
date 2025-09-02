"""
Advanced Voice Integration System for PlannerIA
Système d'interface vocale intelligente avec STT/TTS
"""

from .speech_recognizer import (
    VoiceSpeechRecognizer,
    VoiceCommand,
    RecognitionResult,
    get_voice_recognizer,
    start_voice_recognition,
    stop_voice_recognition
)

from .text_to_speech import (
    VoiceTextToSpeech,
    SpeechConfig,
    get_voice_synthesizer,
    speak_text,
    speak_analysis_result
)

from .voice_processor import (
    VoiceCommandProcessor,
    CommandIntent,
    ProcessingResult,
    get_voice_processor,
    process_voice_command
)

from .voice_ui import (
    render_voice_interface,
    render_voice_floating_controls,
    is_voice_available,
    get_voice_stats
)

__all__ = [
    'VoiceSpeechRecognizer',
    'VoiceCommand',
    'RecognitionResult',
    'VoiceTextToSpeech',
    'SpeechConfig', 
    'VoiceCommandProcessor',
    'CommandIntent',
    'ProcessingResult',
    'get_voice_recognizer',
    'start_voice_recognition',
    'stop_voice_recognition',
    'get_voice_synthesizer',
    'speak_text',
    'speak_analysis_result',
    'get_voice_processor',
    'process_voice_command',
    'render_voice_interface',
    'render_voice_floating_controls',
    'is_voice_available',
    'get_voice_stats'
]