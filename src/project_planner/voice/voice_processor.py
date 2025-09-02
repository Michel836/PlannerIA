"""
Advanced Voice Command Processor for PlannerIA
Processeur intelligent de commandes vocales avec NLP
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class CommandIntent(Enum):
    # Génération de plans
    GENERATE_PLAN = "generate_plan"
    CREATE_PROJECT = "create_project" 
    NEW_PLAN = "new_plan"
    
    # Navigation
    SHOW_ANALYTICS = "show_analytics"
    SHOW_DASHBOARD = "show_dashboard"
    SHOW_MONITORING = "show_monitoring"
    SHOW_COMPARISON = "show_comparison"
    
    # Actions sur les projets
    EXPORT_PDF = "export_pdf"
    EXPORT_CSV = "export_csv"
    SAVE_PROJECT = "save_project"
    
    # Surveillance
    START_MONITORING = "start_monitoring"
    STOP_MONITORING = "stop_monitoring"
    GET_ALERTS = "get_alerts"
    
    # Informations
    GET_STATUS = "get_status"
    GET_HELP = "get_help"
    READ_RESULTS = "read_results"
    
    # Contrôles système
    STOP_VOICE = "stop_voice"
    START_VOICE = "start_voice"
    CLEAR_SCREEN = "clear_screen"
    
    # Inconnu
    UNKNOWN = "unknown"

@dataclass
class ProcessingResult:
    success: bool
    intent: CommandIntent
    confidence: float
    parameters: Dict[str, Any]
    action_taken: Optional[str] = None
    response_text: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'intent': self.intent.value
        }

class VoiceCommandProcessor:
    """Processeur intelligent de commandes vocales"""
    
    def __init__(self):
        self.command_patterns = self._build_command_patterns()
        self.action_handlers: Dict[CommandIntent, Callable] = {}
        self.context_history: List[Dict[str, Any]] = []
        
        # Statistiques
        self.stats = {
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'intent_distribution': {},
            'last_command': None
        }
        
        # Configuration NLP
        self.confidence_threshold = 0.4  # Seuil plus bas pour accepter plus de variations
        self.max_history = 50
        
    def _build_command_patterns(self) -> Dict[CommandIntent, List[str]]:
        """Construire les patterns de reconnaissance de commandes"""
        
        return {
            # Génération de plans
            CommandIntent.GENERATE_PLAN: [
                r"génère?\s+(?:un\s+)?plan\s*(?:pour\s+(.+))?",
                r"crée?\s+(?:un\s+)?plan\s*(?:pour\s+(.+))?",
                r"nouveau\s+plan\s*(?:pour\s+(.+))?",
                r"planifie?\s*(.+)?",
                r"lance?\s+(?:la\s+)?planification\s*(?:pour\s+(.+))?",
                r"je\s+(?:veux|souhaite|voudrais|aimerais)\s+(?:faire|créer|réaliser|développer)\s+(?:un\s+)?(.+)",
                r"j'ai\s+besoin\s+(?:d'un\s+plan\s+pour\s+)?(.+)",
                r"peux-tu\s+(?:créer|faire|générer)\s+(?:un\s+plan\s+pour\s+)?(.+)",
                r"aide-moi\s+(?:à\s+)?(?:planifier|créer|faire)\s+(.+)",
                r"comment\s+(?:faire|créer|développer)\s+(.+)"
            ],
            
            CommandIntent.CREATE_PROJECT: [
                r"crée?\s+(?:un\s+)?projet\s*(?:(.+))?",
                r"nouveau\s+projet\s*(?:(.+))?",
                r"démarre?\s+(?:un\s+)?projet\s*(?:(.+))?",
                r"je\s+(?:veux|souhaite|voudrais)\s+(?:créer|faire|démarrer)\s+(?:un\s+)?projet\s*(.+)?",
                r"réalise?\s+(?:un\s+)?projet\s*(?:pour\s+)?(.+)?",
                r"lance?\s+(?:un\s+)?projet\s*(.+)?"
            ],
            
            # Navigation
            CommandIntent.SHOW_ANALYTICS: [
                r"montre?\s+(?:les\s+)?(?:analyses?|analytics?)",
                r"affiche?\s+(?:les\s+)?(?:analyses?|analytics?)",
                r"va\s+(?:aux?\s+)?(?:analyses?|analytics?)",
                r"ouvre?\s+(?:les\s+)?(?:analyses?|analytics?)"
            ],
            
            CommandIntent.SHOW_DASHBOARD: [
                r"montre?\s+(?:le\s+)?dashboard",
                r"affiche?\s+(?:le\s+)?tableau\s+de\s+bord",
                r"va\s+au\s+dashboard",
                r"retour\s+(?:au\s+)?dashboard"
            ],
            
            CommandIntent.SHOW_MONITORING: [
                r"montre?\s+(?:la\s+)?surveillance",
                r"affiche?\s+(?:le\s+)?monitoring",
                r"va\s+(?:à\s+la\s+)?surveillance",
                r"ouvre?\s+(?:le\s+)?monitoring"
            ],
            
            CommandIntent.SHOW_COMPARISON: [
                r"montre?\s+(?:les?\s+)?comparaisons?",
                r"affiche?\s+(?:les?\s+)?comparaisons?",
                r"compare?\s+(?:les?\s+)?projets?",
                r"va\s+aux?\s+comparaisons?"
            ],
            
            # Actions
            CommandIntent.EXPORT_PDF: [
                r"exporte?\s+(?:en\s+)?(?:pdf|PDF)",
                r"génère?\s+(?:un\s+)?(?:pdf|PDF)",
                r"sauvegarde?\s+(?:en\s+)?(?:pdf|PDF)",
                r"crée?\s+(?:un\s+)?rapport\s+(?:pdf|PDF)?"
            ],
            
            CommandIntent.EXPORT_CSV: [
                r"exporte?\s+(?:en\s+)?(?:csv|CSV)",
                r"génère?\s+(?:un\s+)?(?:csv|CSV)",
                r"sauvegarde?\s+(?:en\s+)?(?:csv|CSV)",
                r"données?\s+(?:csv|CSV)"
            ],
            
            # Surveillance
            CommandIntent.START_MONITORING: [
                r"démarre?\s+(?:la\s+)?surveillance",
                r"active?\s+(?:le\s+)?monitoring",
                r"lance?\s+(?:la\s+)?surveillance",
                r"commence?\s+(?:le\s+)?monitoring"
            ],
            
            CommandIntent.STOP_MONITORING: [
                r"arrête?\s+(?:la\s+)?surveillance",
                r"désactive?\s+(?:le\s+)?monitoring",
                r"stoppe?\s+(?:la\s+)?surveillance",
                r"finis?\s+(?:le\s+)?monitoring"
            ],
            
            CommandIntent.GET_ALERTS: [
                r"montre?\s+(?:les\s+)?alertes?",
                r"affiche?\s+(?:les\s+)?alertes?",
                r"quelles?\s+alertes?",
                r"y\s+a-t-il\s+des\s+alertes?"
            ],
            
            # Informations
            CommandIntent.GET_STATUS: [
                r"(?:quel\s+est\s+le\s+)?(?:statut|état)",
                r"comment\s+(?:ça\s+)?va",
                r"tout\s+va\s+bien",
                r"status"
            ],
            
            CommandIntent.GET_HELP: [
                r"aide",
                r"help",
                r"comment\s+(?:faire|utiliser)",
                r"que\s+puis-je\s+(?:faire|dire)"
            ],
            
            CommandIntent.READ_RESULTS: [
                r"lis?(?:-moi)?\s+(?:les\s+)?résultats?",
                r"raconte?(?:-moi)?\s+(?:les\s+)?résultats?",
                r"dis?(?:-moi)?\s+(?:les\s+)?résultats?",
                r"explique?(?:-moi)?\s+(?:l[ae]\s+)?(?:analyse|résultat)"
            ],
            
            # Contrôles
            CommandIntent.STOP_VOICE: [
                r"arrête?\s+(?:la\s+)?voix",
                r"stop\s+(?:voice|voix)",
                r"silence",
                r"tais?-toi"
            ],
            
            CommandIntent.START_VOICE: [
                r"démarre?\s+(?:la\s+)?voix",
                r"active?\s+(?:la\s+)?voix",
                r"start\s+(?:voice|voix)",
                r"écoute?(?:-moi)?"
            ],
            
            CommandIntent.CLEAR_SCREEN: [
                r"efface?\s+(?:l[ae]\s+)?écran",
                r"nettoie?\s+(?:l[ae]\s+)?écran",
                r"clear",
                r"vide?\s+(?:l[ae]\s+)?écran"
            ]
        }
        
    def register_action_handler(self, intent: CommandIntent, handler: Callable):
        """Enregistrer un gestionnaire d'action pour une intention"""
        self.action_handlers[intent] = handler
        logger.debug(f"Handler enregistré pour {intent.value}")
        
    def process_command(self, command_text: str) -> ProcessingResult:
        """Traiter une commande vocale"""
        if not command_text or not command_text.strip():
            return ProcessingResult(
                success=False,
                intent=CommandIntent.UNKNOWN,
                confidence=0.0,
                parameters={},
                error="Commande vide"
            )
            
        # Nettoyer et normaliser le texte
        cleaned_text = self._clean_text(command_text)
        
        # Analyser l'intention et extraire les paramètres
        intent, confidence, parameters = self._analyze_intent(cleaned_text)
        
        # Mise à jour statistiques
        self._update_stats(intent, success=confidence >= self.confidence_threshold)
        
        # Exécuter l'action si confiance suffisante
        if confidence >= self.confidence_threshold:
            result = self._execute_action(intent, parameters, cleaned_text)
        else:
            result = ProcessingResult(
                success=False,
                intent=intent,
                confidence=confidence,
                parameters=parameters,
                error=f"Confiance insuffisante: {confidence:.2f} < {self.confidence_threshold}"
            )
            
        # Ajouter au contexte
        self._add_to_context({
            'command': command_text,
            'intent': intent.value,
            'confidence': confidence,
            'success': result.success,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
        
    def _clean_text(self, text: str) -> str:
        """Nettoyer et normaliser le texte de commande"""
        # Minuscules
        text = text.lower().strip()
        
        # Normaliser les contractions françaises
        contractions = {
            "j'ai": "j'ai",
            "j'aimerais": "j'aimerais", 
            "d'un": "d'un",
            "c'est": "c'est",
            "qu'est": "qu'est",
            "n'ai": "n'ai"
        }
        
        # Remplacements courants
        replacements = {
            'planneria': 'plannerai',
            'plannerai': 'plannerai',
            'pdf': 'PDF',
            'csv': 'CSV',
            'réaliser': 'réaliser',
            'créer': 'créer'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Supprimer ponctuation excessive mais garder apostrophes importantes
        text = re.sub(r'[!?.,;:]+', ' ', text)
        
        # Normaliser espaces
        text = re.sub(r'\s+', ' ', text)
            
        return text.strip()
        
    def _analyze_intent(self, text: str) -> Tuple[CommandIntent, float, Dict[str, Any]]:
        """Analyser l'intention d'une commande"""
        
        best_intent = CommandIntent.UNKNOWN
        best_confidence = 0.0
        best_parameters = {}
        
        # Tester chaque pattern d'intention
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        # Calculer score de confiance basé sur la qualité du match
                        confidence = self._calculate_confidence(match, text, pattern)
                        
                        if confidence > best_confidence:
                            best_intent = intent
                            best_confidence = confidence
                            best_parameters = self._extract_parameters(match, intent)
                            
                except Exception as e:
                    logger.error(f"Erreur analyse pattern {pattern}: {e}")
                    continue
                    
        return best_intent, best_confidence, best_parameters
        
    def _calculate_confidence(self, match, text: str, pattern: str) -> float:
        """Calculer le score de confiance d'un match"""
        
        # Score de base selon la longueur du match
        match_length = len(match.group(0))
        text_length = len(text)
        coverage = match_length / text_length
        
        # Bonus si match complet ou presque
        if coverage >= 0.7:
            base_score = 0.9
        elif coverage >= 0.5:
            base_score = 0.8
        elif coverage >= 0.3:
            base_score = 0.7
        elif coverage >= 0.2:
            base_score = 0.6
        else:
            base_score = 0.5  # Score minimum plus généreux
            
        # Bonus important pour groupes capturés (paramètres détaillés)
        if match.groups():
            params = [g for g in match.groups() if g and g.strip()]
            base_score += len(params) * 0.1  # Bonus plus important
            
        # Bonus pour mots-clés importants dans le pattern
        key_words = ['souhaite', 'veux', 'voudrais', 'aimerais', 'réaliser', 'créer', 'faire', 'projet', 'plan']
        found_keywords = sum(1 for word in key_words if word in text)
        base_score += found_keywords * 0.05
        
        # Malus réduit pour texte non reconnu
        remaining_text = text.replace(match.group(0), '').strip()
        if len(remaining_text) > len(text) * 0.4:  # Seuil plus tolérant
            base_score -= 0.05  # Malus réduit
            
        return min(1.0, max(0.3, base_score))  # Score minimum de 0.3
        
    def _extract_parameters(self, match, intent: CommandIntent) -> Dict[str, Any]:
        """Extraire les paramètres d'un match"""
        parameters = {}
        
        if match.groups():
            groups = [g.strip() if g else '' for g in match.groups()]
            
            # Extraction spécifique par intention
            if intent in [CommandIntent.GENERATE_PLAN, CommandIntent.CREATE_PROJECT]:
                if groups[0]:
                    parameters['project_description'] = groups[0]
                    
            elif intent == CommandIntent.EXPORT_PDF:
                parameters['format'] = 'pdf'
                
            elif intent == CommandIntent.EXPORT_CSV:
                parameters['format'] = 'csv'
                
        return parameters
        
    def _execute_action(self, intent: CommandIntent, parameters: Dict[str, Any], original_text: str) -> ProcessingResult:
        """Exécuter une action basée sur l'intention"""
        
        # Vérifier si un handler est enregistré
        if intent in self.action_handlers:
            try:
                handler = self.action_handlers[intent]
                result = handler(parameters, original_text)
                
                return ProcessingResult(
                    success=True,
                    intent=intent,
                    confidence=1.0,
                    parameters=parameters,
                    action_taken=result.get('action', 'executed'),
                    response_text=result.get('response', f"Commande {intent.value} exécutée")
                )
                
            except Exception as e:
                logger.error(f"Erreur exécution handler {intent.value}: {e}")
                return ProcessingResult(
                    success=False,
                    intent=intent,
                    confidence=0.8,
                    parameters=parameters,
                    error=f"Erreur exécution: {e}"
                )
        else:
            # Pas de handler - retourner intention détectée
            return ProcessingResult(
                success=True,
                intent=intent,
                confidence=0.8,
                parameters=parameters,
                action_taken="detected",
                response_text=f"Intention {intent.value} détectée mais pas de handler configuré"
            )
            
    def _add_to_context(self, context_entry: Dict[str, Any]):
        """Ajouter une entrée au contexte historique"""
        self.context_history.append(context_entry)
        
        # Limiter la taille de l'historique
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]
            
    def _update_stats(self, intent: CommandIntent, success: bool):
        """Mettre à jour les statistiques"""
        self.stats['total_commands'] += 1
        
        if success:
            self.stats['successful_commands'] += 1
        else:
            self.stats['failed_commands'] += 1
            
        # Distribution des intentions
        intent_name = intent.value
        if intent_name not in self.stats['intent_distribution']:
            self.stats['intent_distribution'][intent_name] = 0
        self.stats['intent_distribution'][intent_name] += 1
        
        self.stats['last_command'] = {
            'intent': intent_name,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de traitement"""
        total = self.stats['total_commands']
        success_rate = (
            self.stats['successful_commands'] / total
            if total > 0 else 0
        )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'context_history_size': len(self.context_history)
        }
        
    def get_supported_commands(self) -> Dict[str, List[str]]:
        """Obtenir la liste des commandes supportées"""
        supported = {}
        
        for intent, patterns in self.command_patterns.items():
            # Convertir patterns regex en exemples lisibles
            examples = []
            for pattern in patterns[:3]:  # Limiter à 3 exemples par intention
                # Simplifier le pattern pour l'exemple
                example = pattern.replace(r'\s+', ' ').replace('?', '').replace(r'(.+)', '[description]')
                example = re.sub(r'[\\(){}[\]^$*+?.|]', '', example)
                examples.append(example.strip())
                
            supported[intent.value] = examples
            
        return supported
        
    def test_command_recognition(self, test_commands: List[str]) -> List[Dict[str, Any]]:
        """Tester la reconnaissance sur une liste de commandes"""
        results = []
        
        for command in test_commands:
            result = self.process_command(command)
            results.append({
                'command': command,
                'intent': result.intent.value,
                'confidence': result.confidence,
                'success': result.success,
                'parameters': result.parameters
            })
            
        return results

# Instance globale
global_voice_processor: Optional[VoiceCommandProcessor] = None

def get_voice_processor() -> VoiceCommandProcessor:
    """Obtenir l'instance globale du processeur"""
    global global_voice_processor
    
    if global_voice_processor is None:
        global_voice_processor = VoiceCommandProcessor()
        
    return global_voice_processor

def process_voice_command(command_text: str) -> ProcessingResult:
    """Traiter une commande vocale (fonction utilitaire)"""
    processor = get_voice_processor()
    return processor.process_command(command_text)