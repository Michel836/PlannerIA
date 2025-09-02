"""
🤖 Assistant IA Conversationnel - PlannerIA
Assistant intelligent pour analyse de projets, recommandations et optimisations
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime
import numpy as np
from .predictive_engine import ai_predictor, ProjectDomain


class ConversationType(Enum):
    PROJECT_ANALYSIS = "analyse_projet"
    OPTIMIZATION = "optimisation"
    RISK_ASSESSMENT = "evaluation_risques"
    TEAM_PLANNING = "planification_equipe"
    BUDGET_PLANNING = "planification_budget"
    GENERAL_ADVICE = "conseil_general"


@dataclass
class ConversationContext:
    """Contexte de la conversation"""
    user_id: str
    session_id: str
    project_context: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_preferences is None:
            self.user_preferences = {}


class AIConversationalAssistant:
    """Assistant IA conversationnel pour PlannerIA"""
    
    def __init__(self):
        self.context = None
        self.intent_patterns = self._initialize_intent_patterns()
        self.response_templates = self._initialize_response_templates()
        self.knowledge_base = self._initialize_knowledge_base()
        
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialise les patterns de reconnaissance d'intention"""
        return {
            ConversationType.PROJECT_ANALYSIS.value: [
                r"analys[er|e] (?:mon|le) projet",
                r"que penses-tu de (?:mon|ce) projet",
                r"évaluer? (?:mon|le) projet",
                r"faisabilité d[ue] projet",
                r"chances de succès",
                r"projet .* avec .* développeurs?"
            ],
            ConversationType.OPTIMIZATION.value: [
                r"optimis[er|ation]",
                r"améliorer (?:mon|le) projet",
                r"comment faire mieux",
                r"réduire les coûts?",
                r"accélérer le développement",
                r"plus efficace"
            ],
            ConversationType.RISK_ASSESSMENT.value: [
                r"risques?",
                r"dangers?",
                r"problèmes? potentiels?",
                r"qu'est-ce qui peut mal se passer",
                r"mitigation",
                r"sécuriser le projet"
            ],
            ConversationType.TEAM_PLANNING.value: [
                r"équipe",
                r"développeurs?",
                r"ressources? humaines?",
                r"combien de personnes?",
                r"compétences? nécessaires?",
                r"recrutement"
            ],
            ConversationType.BUDGET_PLANNING.value: [
                r"budget",
                r"coût",
                r"prix",
                r"combien ça coûte",
                r"financement",
                r"rentabilité"
            ]
        }
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialise les templates de réponse"""
        return {
            'greeting': [
                "👋 Salut ! Je suis votre assistant IA PlannerIA. Comment puis-je vous aider avec votre projet ?",
                "🤖 Bonjour ! Prêt à optimiser votre projet ensemble ?",
                "✨ Hello ! Votre assistant intelligent PlannerIA à votre service !"
            ],
            'analysis_intro': [
                "🔍 Parfait ! Laissez-moi analyser votre projet...",
                "📊 Excellente question ! Je vais examiner tous les aspects...",
                "🎯 Intéressant ! Voici mon analyse détaillée..."
            ],
            'optimization_intro': [
                "⚡ Je vois plusieurs opportunités d'optimisation...",
                "🚀 Voici comment améliorer votre projet...",
                "💡 J'ai quelques suggestions brillantes pour vous..."
            ],
            'risk_intro': [
                "⚠️ Voici les risques que j'identifie...",
                "🛡️ Parlons des points de vigilance...",
                "🔍 Mon analyse des risques révèle..."
            ]
        }
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialise la base de connaissances"""
        return {
            'best_practices': {
                'agile': {
                    'description': "Méthodologie agile pour projets adaptatifs",
                    'benefits': ["Flexibilité", "Feedback rapide", "Livraisons fréquentes"],
                    'when_to_use': "Projets avec requirements évolutifs"
                },
                'devops': {
                    'description': "Intégration développement et opérations",
                    'benefits': ["Déploiements rapides", "Qualité", "Automatisation"],
                    'when_to_use': "Projets nécessitant des déploiements fréquents"
                },
                'microservices': {
                    'description': "Architecture en services distribués",
                    'benefits': ["Scalabilité", "Indépendance", "Technologie diverse"],
                    'when_to_use': "Grandes applications complexes"
                }
            },
            'technologies': {
                'react': {'type': 'frontend', 'complexity': 0.6, 'learning_curve': 0.7},
                'vue': {'type': 'frontend', 'complexity': 0.5, 'learning_curve': 0.5},
                'angular': {'type': 'frontend', 'complexity': 0.8, 'learning_curve': 0.9},
                'node.js': {'type': 'backend', 'complexity': 0.5, 'learning_curve': 0.6},
                'python': {'type': 'backend', 'complexity': 0.4, 'learning_curve': 0.4},
                'java': {'type': 'backend', 'complexity': 0.7, 'learning_curve': 0.8}
            },
            'domain_expertise': {
                'ecommerce': {
                    'key_features': ['Catalogue produits', 'Panier', 'Paiement', 'Gestion stocks'],
                    'challenges': ['Scalabilité', 'Sécurité paiements', 'UX mobile'],
                    'typical_duration': '4-8 mois',
                    'team_size': '5-10 personnes'
                },
                'fintech': {
                    'key_features': ['Transactions', 'KYC', 'Conformité', 'Sécurité'],
                    'challenges': ['Régulation', 'Sécurité', 'Performance'],
                    'typical_duration': '8-15 mois',
                    'team_size': '8-15 personnes'
                }
            }
        }
    
    async def process_message(self, message: str, context: ConversationContext) -> str:
        """Traite un message utilisateur et génère une réponse intelligente"""
        
        self.context = context
        
        # Nettoyage et analyse du message
        cleaned_message = self._clean_message(message)
        intent = self._detect_intent(cleaned_message)
        entities = self._extract_entities(cleaned_message)
        
        # Mise à jour du contexte
        self._update_conversation_context(message, intent, entities)
        
        # Génération de la réponse
        response = await self._generate_response(intent, entities, cleaned_message)
        
        return response
    
    def _clean_message(self, message: str) -> str:
        """Nettoie le message utilisateur"""
        # Suppression des caractères spéciaux, normalisation
        cleaned = message.lower().strip()
        # Suppression des mots vides si nécessaire
        return cleaned
    
    def _detect_intent(self, message: str) -> ConversationType:
        """Détecte l'intention de l'utilisateur"""
        
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            
            if score > 0:
                intent_scores[intent_type] = score
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            return ConversationType(best_intent)
        
        return ConversationType.GENERAL_ADVICE
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extrait les entités du message"""
        entities = {}
        
        # Extraction de chiffres (budget, équipe, durée)
        numbers = re.findall(r'\b\d+\b', message)
        
        # Budget (euros, dollars)
        budget_patterns = [
            r'(\d+)\s*(?:euros?|€)',
            r'(\d+)\s*(?:dollars?|\$)',
            r'budget.*?(\d+)',
            r'(\d+)\s*k€?'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                budget = int(match.group(1))
                if 'k' in match.group(0).lower():
                    budget *= 1000
                entities['budget'] = budget
                break
        
        # Taille d'équipe
        team_patterns = [
            r'(\d+)\s*(?:développeurs?|devs?|personnes?)',
            r'équipe.*?(\d+)',
            r'(\d+)\s*(?:membres?|gens?)'
        ]
        
        for pattern in team_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entities['team_size'] = int(match.group(1))
                break
        
        # Durée
        duration_patterns = [
            r'(\d+)\s*(?:mois|months?)',
            r'(\d+)\s*(?:semaines?|weeks?)',
            r'(\d+)\s*(?:jours?|days?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                duration = int(match.group(1))
                if 'semaine' in match.group(0) or 'week' in match.group(0):
                    duration *= 7  # Convertir en jours
                elif 'mois' in match.group(0) or 'month' in match.group(0):
                    duration *= 30  # Convertir en jours
                entities['duration'] = duration
                break
        
        # Technologies mentionnées
        technologies = []
        tech_keywords = self.knowledge_base['technologies'].keys()
        for tech in tech_keywords:
            if tech.lower() in message.lower():
                technologies.append(tech)
        
        if technologies:
            entities['technologies'] = technologies
        
        # Domaine du projet
        domain_keywords = {
            'ecommerce': ['e-commerce', 'boutique', 'vente en ligne', 'shop', 'marketplace'],
            'fintech': ['banque', 'finance', 'paiement', 'crypto', 'blockchain'],
            'healthcare': ['santé', 'médical', 'hôpital', 'patient'],
            'mobile': ['mobile', 'app', 'ios', 'android', 'smartphone']
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in message.lower():
                    entities['domain'] = domain
                    break
            if 'domain' in entities:
                break
        
        return entities
    
    def _update_conversation_context(self, message: str, intent: ConversationType, entities: Dict[str, Any]):
        """Met à jour le contexte de conversation"""
        if self.context:
            self.context.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'intent': intent.value,
                'entities': entities
            })
            
            # Mise à jour du contexte projet avec les nouvelles entités
            if self.context.project_context is None:
                self.context.project_context = {}
            
            self.context.project_context.update(entities)
    
    async def _generate_response(self, intent: ConversationType, entities: Dict[str, Any], message: str) -> str:
        """Génère une réponse intelligente basée sur l'intention"""
        
        try:
            if intent == ConversationType.PROJECT_ANALYSIS:
                return await self._generate_project_analysis(entities)
            elif intent == ConversationType.OPTIMIZATION:
                return await self._generate_optimization_advice(entities)
            elif intent == ConversationType.RISK_ASSESSMENT:
                return await self._generate_risk_assessment(entities)
            elif intent == ConversationType.TEAM_PLANNING:
                return await self._generate_team_advice(entities)
            elif intent == ConversationType.BUDGET_PLANNING:
                return await self._generate_budget_advice(entities)
            else:
                return await self._generate_general_response(message)
                
        except Exception as e:
            return f"🤖 Désolé, j'ai rencontré un petit problème. Pouvez-vous reformuler votre question ?"
    
    async def _generate_project_analysis(self, entities: Dict[str, Any]) -> str:
        """Génère une analyse de projet complète"""
        
        # Construction des données projet pour l'IA prédictive
        project_data = {
            'domain': entities.get('domain', 'web_app'),
            'team_size': entities.get('team_size', 5),
            'budget': entities.get('budget', 100000),
            'technology_stack': entities.get('technologies', ['react', 'node.js']),
            'complexity': 'medium',
            'team_experience': 'mixed'
        }
        
        # Utilisation de l'IA prédictive
        prediction = await ai_predictor.predict_project_outcome(project_data)
        
        # Construction de la réponse
        intro = np.random.choice(self.response_templates['analysis_intro'])
        
        response = f"""{intro}

🎯 **Analyse de votre projet {project_data['domain'].upper()}**

📊 **Prédictions IA:**
• **Durée estimée**: {prediction.predicted_duration:.0f} jours
• **Coût prévu**: {prediction.predicted_cost:,.0f}€
• **Probabilité de succès**: {prediction.success_probability:.1%}

⚡ **Facteurs de risque:**"""
        
        for risk_name, risk_score in prediction.risk_factors.items():
            risk_emoji = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.4 else "🟢"
            risk_label = risk_name.replace('_', ' ').title()
            response += f"\n• {risk_emoji} {risk_label}: {risk_score:.1%}"
        
        response += f"\n\n💡 **Mes recommandations:**"
        for i, rec in enumerate(prediction.recommendations[:3], 1):
            response += f"\n{i}. {rec}"
        
        # Ajout d'insights domaine
        domain_insights = ai_predictor.get_domain_insights(project_data['domain'])
        if 'error' not in domain_insights:
            response += f"""

🏆 **Insights domaine {project_data['domain'].upper()}:**
• Durée type: {domain_insights['average_duration_multiplier']:.1f}x la baseline
• Taux de succès moyen: {domain_insights['average_success_rate']:.1%}
• Risques principaux: {', '.join(domain_insights['top_risks'][:3])}"""
        
        return response
    
    async def _generate_optimization_advice(self, entities: Dict[str, Any]) -> str:
        """Génère des conseils d'optimisation"""
        
        intro = np.random.choice(self.response_templates['optimization_intro'])
        
        optimizations = []
        
        # Optimisations basées sur les entités détectées
        if entities.get('team_size', 0) > 8:
            optimizations.append("🏗️ **Équipes autonomes**: Divisez en équipes plus petites (3-5 personnes) pour plus d'efficacité")
        
        if entities.get('budget') and entities.get('budget') < 50000:
            optimizations.append("💡 **Solutions Low-Code**: Explorez des plateformes no-code/low-code pour réduire les coûts")
        
        if 'technologies' in entities:
            tech_complexity = sum([self.knowledge_base['technologies'].get(tech, {}).get('complexity', 0.5) 
                                 for tech in entities['technologies']])
            if tech_complexity > 1.5:
                optimizations.append("⚡ **Stack simplifiée**: Votre stack est complexe, considérez des alternatives plus simples")
        
        # Optimisations génériques
        general_optimizations = [
            "🎯 **MVP First**: Commencez par un MVP pour valider rapidement",
            "🔄 **Sprints courts**: Adoptez des sprints de 2 semaines maximum",
            "🤖 **Automatisation**: Investissez dans les tests et déploiements automatiques",
            "📈 **Métriques**: Mettez en place un monitoring dès le début",
            "👥 **Communication**: Utilisez des outils collaboratifs modernes"
        ]
        
        optimizations.extend(general_optimizations[:3])
        
        response = f"{intro}\n\n"
        for i, opt in enumerate(optimizations, 1):
            response += f"{i}. {opt}\n"
        
        return response
    
    async def _generate_risk_assessment(self, entities: Dict[str, Any]) -> str:
        """Génère une évaluation des risques"""
        
        intro = np.random.choice(self.response_templates['risk_intro'])
        
        risks = []
        
        # Risques basés sur le contexte
        if entities.get('team_size', 0) < 3:
            risks.append("⚠️ **Équipe réduite**: Risque de surcharge et de manque de compétences croisées")
        
        if entities.get('team_size', 0) > 10:
            risks.append("⚠️ **Grande équipe**: Risque de problèmes de communication et coordination")
        
        if entities.get('budget', 100000) < 30000:
            risks.append("💸 **Budget serré**: Risque de compromis sur la qualité ou les fonctionnalités")
        
        if 'domain' in entities:
            domain = entities['domain']
            domain_info = self.knowledge_base['domain_expertise'].get(domain, {})
            domain_challenges = domain_info.get('challenges', [])
            for challenge in domain_challenges[:2]:
                risks.append(f"🎯 **Défi {domain}**: {challenge}")
        
        # Risques techniques
        if 'technologies' in entities:
            cutting_edge_techs = [tech for tech in entities['technologies'] 
                                if self.knowledge_base['technologies'].get(tech, {}).get('complexity', 0.5) > 0.7]
            if cutting_edge_techs:
                risks.append(f"🔧 **Technos complexes**: {', '.join(cutting_edge_techs)} nécessitent expertise avancée")
        
        # Mitigations
        mitigations = [
            "🛡️ **Tests automatisés**: Implémentez une couverture de tests robuste",
            "📋 **Documentation**: Maintenez une documentation à jour",
            "🔄 **Revues de code**: Mettez en place des process de revue systématiques",
            "💾 **Backup plan**: Préparez des solutions de fallback",
            "📊 **Monitoring**: Surveillez les métriques critiques en continu"
        ]
        
        response = f"{intro}\n\n**⚠️ Risques identifiés:**\n"
        for i, risk in enumerate(risks[:4], 1):
            response += f"{i}. {risk}\n"
        
        response += f"\n**🛡️ Stratégies de mitigation:**\n"
        for i, mitigation in enumerate(mitigations[:3], 1):
            response += f"{i}. {mitigation}\n"
        
        return response
    
    async def _generate_team_advice(self, entities: Dict[str, Any]) -> str:
        """Génère des conseils sur l'équipe"""
        
        team_size = entities.get('team_size', 5)
        domain = entities.get('domain', 'web_app')
        technologies = entities.get('technologies', [])
        
        response = "👥 **Conseils équipe pour votre projet:**\n\n"
        
        # Analyse taille équipe
        if team_size < 3:
            response += "⚠️ **Équipe petite** (< 3 personnes):\n"
            response += "• Avantage: Communication facile, décisions rapides\n"
            response += "• Risque: Charge de travail élevée, manque de spécialisation\n"
            response += "• Conseil: Priorisez les compétences polyvalentes\n\n"
        elif team_size <= 8:
            response += "✅ **Équipe optimale** (3-8 personnes):\n"
            response += "• Parfait pour la plupart des projets\n"
            response += "• Bonne balance communication/spécialisation\n"
            response += "• Conseil: Définissez clairement les rôles\n\n"
        else:
            response += "⚠️ **Grande équipe** (> 8 personnes):\n"
            response += "• Avantage: Plus de compétences, développement rapide\n"
            response += "• Risque: Coordination complexe, communication difficile\n"
            response += "• Conseil: Divisez en équipes plus petites\n\n"
        
        # Composition recommandée
        response += "🎯 **Composition recommandée:**\n"
        
        domain_recommendations = {
            'ecommerce': ["1 Product Owner", "2-3 Développeurs Full-Stack", "1 Designer UX", "1 DevOps"],
            'fintech': ["1 Product Owner", "2 Développeurs Backend", "1 Frontend", "1 Expert Sécurité", "1 Compliance"],
            'mobile': ["1 Product Owner", "1 Développeur iOS", "1 Développeur Android", "1 Designer", "1 QA"]
        }
        
        recommendations = domain_recommendations.get(domain, 
            ["1 Product Owner", "2 Développeurs Full-Stack", "1 Designer", "1 QA"])
        
        for rec in recommendations:
            response += f"• {rec}\n"
        
        # Conseils technologies
        if technologies:
            response += f"\n🔧 **Pour votre stack ({', '.join(technologies)}):**\n"
            for tech in technologies:
                tech_info = self.knowledge_base['technologies'].get(tech, {})
                if tech_info.get('complexity', 0.5) > 0.7:
                    response += f"• {tech}: Nécessite développeur expérimenté\n"
        
        return response
    
    async def _generate_budget_advice(self, entities: Dict[str, Any]) -> str:
        """Génère des conseils budgétaires"""
        
        budget = entities.get('budget')
        team_size = entities.get('team_size', 5)
        duration = entities.get('duration', 90)  # 3 mois par défaut
        
        response = "💰 **Analyse budgétaire:**\n\n"
        
        if budget:
            # Calcul coût développeur moyen (50k€/an = ~400€/jour)
            daily_rate = 400
            total_dev_days = team_size * duration
            estimated_dev_cost = total_dev_days * daily_rate
            
            response += f"📊 **Votre budget**: {budget:,}€\n"
            response += f"🧮 **Coût développement estimé**: {estimated_dev_cost:,}€\n"
            response += f"📈 **Ratio budget/dev**: {(budget/estimated_dev_cost)*100:.1f}%\n\n"
            
            if budget < estimated_dev_cost * 0.8:
                response += "🔴 **Budget serré** - Considérez:\n"
                response += "• Réduire le scope initial\n"
                response += "• Adopter une approche MVP\n"
                response += "• Explorer les solutions low-code\n"
                response += "• Décaler certaines fonctionnalités\n\n"
            elif budget > estimated_dev_cost * 1.3:
                response += "🟢 **Budget confortable** - Vous pouvez:\n"
                response += "• Investir dans la qualité\n"
                response += "• Ajouter des fonctionnalités premium\n"
                response += "• Renforcer les tests et sécurité\n"
                response += "• Prévoir une marge pour l'évolution\n\n"
            else:
                response += "🟡 **Budget réaliste** - Recommandations:\n"
                response += "• Planifiez avec une marge de 20%\n"
                response += "• Priorisez rigoureusement les features\n"
                response += "• Surveillez l'avancement de près\n\n"
        
        # Répartition budget type
        response += "📋 **Répartition budget recommandée:**\n"
        response += "• 60-70% Développement\n"
        response += "• 10-15% Design/UX\n"
        response += "• 10-15% Infrastructure/DevOps\n"
        response += "• 5-10% Tests/QA\n"
        response += "• 5-10% Marge/Imprévu\n"
        
        return response
    
    async def _generate_general_response(self, message: str) -> str:
        """Génère une réponse générale"""
        
        general_responses = [
            "🤖 Je suis là pour vous aider avec votre projet ! Parlez-moi de votre idée, votre équipe, votre budget ou vos préoccupations.",
            "✨ Excellent ! Je peux vous aider avec l'analyse de projet, l'optimisation, les risques, la planification d'équipe ou le budget. Que souhaitez-vous explorer ?",
            "🎯 Intéressant ! Donnez-moi plus de détails sur votre projet et je pourrai vous fournir une analyse personnalisée."
        ]
        
        return np.random.choice(general_responses)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la conversation"""
        if not self.context or not self.context.conversation_history:
            return {'error': 'Aucune conversation en cours'}
        
        history = self.context.conversation_history
        intents = [entry['intent'] for entry in history]
        entities_mentioned = {}
        
        for entry in history:
            for key, value in entry['entities'].items():
                entities_mentioned[key] = value
        
        return {
            'messages_count': len(history),
            'main_intents': list(set(intents)),
            'project_context': entities_mentioned,
            'last_update': history[-1]['timestamp'] if history else None
        }


# Instance globale
ai_assistant = AIConversationalAssistant()