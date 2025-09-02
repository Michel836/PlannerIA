"""
🎬 Demo Portfolio - Projets exemplaires pour soutenance
Création de projets de démonstration réalistes et impressionnants
"""

import json
import os
from datetime import datetime, timedelta
from project_planner.ai.portfolio_manager import (
    ai_portfolio_manager, ProjectStatus, ProjectPriority, ProjectTemplate,
    ProjectDependency
)

def create_demo_portfolio():
    """Crée un portefeuille de projets exemplaires pour la soutenance"""
    
    print("[DEMO] Creation du portefeuille exemplaire...")
    
    # Projet 1: Application FinTech revolutionnaire (Flagship project)
    project1 = ai_portfolio_manager.create_project_wizard(
        name="NeoBank - Super App Financière",
        description="""
        Application bancaire nouvelle génération avec IA intégrée :
        - Gestion multi-comptes intelligente avec catégorisation automatique
        - Assistant IA personnel pour conseils financiers
        - Trading social avec copy trading et insights communautaires  
        - Paiements instantanés avec QR codes et NFC
        - Crédit scoring en temps réel avec ML avancé
        - Interface conversationnelle pour toutes les opérations
        - Sécurité biométrique multi-facteurs
        
        Cible : 100K utilisateurs la première année
        Révolutionnaire dans le secteur bancaire français
        """,
        template=ProjectTemplate.MOBILE_APP,
        team_size=12,
        budget=850000,
        priority=ProjectPriority.CRITICAL
    )
    
    # Projet 2: Plateforme E-learning avec IA
    project2 = ai_portfolio_manager.create_project_wizard(
        name="EduGenius - Plateforme d'apprentissage adaptatif",
        description="""
        Plateforme e-learning révolutionnaire avec IA générative :
        - Parcours personnalisés adaptatifs basés sur le profil apprenant
        - Génération automatique de contenu avec GPT-4
        - Évaluation intelligente avec feedback personnalisé
        - Réalité virtuelle pour formations immersives
        - Chatbot tuteur disponible 24/7
        - Analytics avancés pour les formateurs
        - Gamification avec achievements et classements
        
        Secteur : Formation professionnelle et universitaire
        Innovation majeure dans l'EdTech française
        """,
        template=ProjectTemplate.WEB_APP,
        team_size=8,
        budget=420000,
        priority=ProjectPriority.HIGH
    )
    
    # Projet 3: Solution IA pour Smart Cities
    project3 = ai_portfolio_manager.create_project_wizard(
        name="CityBrain - Intelligence Urbaine",
        description="""
        Solution IA complète pour la gestion intelligente des villes :
        - Optimisation du trafic en temps réel avec IoT
        - Prédiction et gestion des déchets urbains
        - Monitoring environnemental (air, bruit, énergie)
        - Détection prédictive des incidents urbains
        - Dashboard unifié pour les décideurs municipaux
        - API publique pour développeurs tiers
        - Intégration avec systèmes existants des mairies
        
        Impact : 500K+ citoyens bénéficiaires
        Projet pilote pour 3 métropoles françaises
        """,
        template=ProjectTemplate.AI_PROJECT,
        team_size=6,
        budget=680000,
        priority=ProjectPriority.HIGH
    )
    
    # Projet 4: Marketplace B2B innovant
    project4 = ai_portfolio_manager.create_project_wizard(
        name="TradeConnect - Marketplace B2B Intelligent",
        description="""
        Marketplace nouvelle génération pour le commerce B2B :
        - Matching intelligent fournisseurs-acheteurs avec IA
        - Négociation automatisée de prix avec algorithmes
        - Supply chain transparente avec blockchain
        - Prédiction de demande et optimisation stocks
        - Paiements sécurisés avec escrow intelligent
        - Analytics avancés pour optimisation commerciale
        - Intégration ERP/CRM native
        
        Secteur : Commerce inter-entreprises
        Disruption du marché B2B traditionnel
        """,
        template=ProjectTemplate.WEB_APP,
        team_size=10,
        budget=550000,
        priority=ProjectPriority.MEDIUM
    )
    
    # Projet 5: Campagne Marketing Omnicanal
    project5 = ai_portfolio_manager.create_project_wizard(
        name="Campaign360 - Marketing IA Omnicanal",
        description="""
        Campagne marketing révolutionnaire multi-plateformes :
        - Personnalisation en temps réel basée sur comportement
        - Création automatique de contenu avec IA générative
        - Attribution marketing cross-device avancée
        - A/B testing automatique et optimisation continue
        - Prédiction de lifetime value client
        - Intégration native réseaux sociaux et programmatique
        - ROI tracking granulaire et real-time
        
        Objectif : +300% ROI vs campagnes traditionnelles
        Innovation dans le marketing digital
        """,
        template=ProjectTemplate.MARKETING,
        team_size=7,
        budget=180000,
        priority=ProjectPriority.MEDIUM
    )
    
    # Projet 6: Infrastructure Cloud Native
    project6 = ai_portfolio_manager.create_project_wizard(
        name="CloudForge - Infrastructure Auto-Scalable",
        description="""
        Infrastructure cloud révolutionnaire auto-optimisée :
        - Auto-scaling intelligent basé sur prédictions ML
        - Orchestration multi-cloud avec optimisation coûts
        - Monitoring prédictif avec auto-healing
        - CI/CD pipeline avec tests automatisés IA
        - Sécurité zero-trust avec détection d'anomalies
        - Disaster recovery automatique cross-regions
        - Carbon footprint optimization
        
        Impact : -60% coûts infrastructure, +99.99% uptime
        Révolution dans l'approche DevOps
        """,
        template=ProjectTemplate.AI_PROJECT,
        team_size=5,
        budget=320000,
        priority=ProjectPriority.HIGH
    )
    
    print(f"[OK] [DEMO] {len(ai_portfolio_manager.projects)} projets exemplaires crees")
    
    # Ajout de dépendances réalistes entre projets
    add_realistic_dependencies()
    
    # Simulation de progression des projets
    simulate_project_progress()
    
    # Génération d'insights réalistes
    generate_demo_insights()
    
    return ai_portfolio_manager.projects

def add_realistic_dependencies():
    """Ajoute des dépendances réalistes entre les projets"""
    
    print("[DEMO] Ajout des dependances inter-projets...")
    
    # Récupération des IDs de projets
    projects = list(ai_portfolio_manager.projects.keys())
    
    if len(projects) >= 6:
        # L'infrastructure doit être prête avant les apps
        infrastructure_id = projects[5]  # CloudForge
        neobank_id = projects[0]        # NeoBank
        edugenius_id = projects[1]      # EduGenius
        
        # Dépendances techniques
        ai_portfolio_manager.dependencies.extend([
            ProjectDependency(
                from_project=infrastructure_id,
                to_project=neobank_id,
                dependency_type="technical",
                strength=0.8,
                description="Infrastructure cloud nécessaire pour NeoBank"
            ),
            ProjectDependency(
                from_project=infrastructure_id, 
                to_project=edugenius_id,
                dependency_type="technical",
                strength=0.6,
                description="Plateforme cloud pour EduGenius"
            ),
            ProjectDependency(
                from_project=projects[4],  # Campaign360
                to_project=neobank_id,
                dependency_type="business",
                strength=0.7,
                description="Marketing pour lancement NeoBank"
            )
        ])
    
    print(f"[OK] [DEMO] {len(ai_portfolio_manager.dependencies)} dependances ajoutees")

def simulate_project_progress():
    """Simule une progression réaliste des projets"""
    
    print("[DEMO] Simulation des progressions de projets...")
    
    # Progression differentielle selon les projets
    progress_simulation = [
        65.0,  # NeoBank - En cours de développement avancé
        45.0,  # EduGenius - Phase de prototypage  
        30.0,  # CityBrain - Phase de conception
        80.0,  # TradeConnect - Près du lancement
        90.0,  # Campaign360 - En finalisation
        55.0   # CloudForge - Infrastructure en cours
    ]
    
    # Health scores réalistes
    health_simulation = [
        85.0,  # NeoBank - Quelques défis techniques
        92.0,  # EduGenius - Très bon déroulement
        78.0,  # CityBrain - Complexité réglementaire
        88.0,  # TradeConnect - Sur les rails
        95.0,  # Campaign360 - Excellent
        82.0   # CloudForge - Migration complexe
    ]
    
    projects = list(ai_portfolio_manager.projects.values())
    
    for i, project in enumerate(projects):
        if i < len(progress_simulation):
            project.progress = progress_simulation[i]
            project.health_score = health_simulation[i]
            
            # Mise à jour du statut selon progression
            if project.progress >= 90:
                project.status = ProjectStatus.COMPLETED if project.progress == 100 else ProjectStatus.IN_PROGRESS
            elif project.progress >= 20:
                project.status = ProjectStatus.IN_PROGRESS
            else:
                project.status = ProjectStatus.PLANNING
    
    print("[OK] [DEMO] Progressions simulees avec realisme")

def generate_demo_insights():
    """Génère des insights et analytics impressionnants"""
    
    print("[DEMO] Generation d'insights pour demonstration...")
    
    # Mise a jour des metriques portfolio
    ai_portfolio_manager._update_portfolio_metrics()
    
    # Cache des analytics pour performance
    demo_analytics = {
        "success_rate": 87.5,  # Taux de succès élevé
        "roi_projection": 245.0,  # ROI projecté impressionnant
        "innovation_index": 9.2,  # Score d'innovation élevé
        "market_impact": "Disruptive",
        "competitive_advantage": "Strong",
        "risk_level": "Medium-Low"
    }
    
    ai_portfolio_manager.analytics_cache.update(demo_analytics)
    
    print("[OK] [DEMO] Insights de demonstration generes")

def get_demo_presentation_data():
    """Retourne les données formatées pour présentation"""
    
    portfolio_summary = {
        "total_projects": len(ai_portfolio_manager.projects),
        "total_investment": sum(p.budget for p in ai_portfolio_manager.projects.values()),
        "avg_progress": sum(p.progress for p in ai_portfolio_manager.projects.values()) / len(ai_portfolio_manager.projects),
        "team_members": sum(p.team_size for p in ai_portfolio_manager.projects.values()),
        "domains_covered": ["FinTech", "EdTech", "Smart Cities", "B2B Commerce", "Marketing", "Infrastructure"],
        "innovation_areas": ["IA Générative", "Machine Learning", "Blockchain", "IoT", "Réalité Virtuelle"],
        "projected_roi": "245%",
        "market_impact": "Disruptive Innovation",
        "expected_users": "600K+ utilisateurs touchés"
    }
    
    return portfolio_summary

def save_demo_portfolio():
    """Sauvegarde le portfolio de démonstration"""
    
    demo_data = {
        "projects": {},
        "created_at": datetime.now().isoformat(),
        "purpose": "Soutenance Demo Portfolio",
        "summary": get_demo_presentation_data()
    }
    
    # Sérialisation des projets
    for project_id, project in ai_portfolio_manager.projects.items():
        demo_data["projects"][project_id] = {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status.value,
            "priority": project.priority.value,
            "template": project.template.value,
            "budget": project.budget,
            "team_size": project.team_size,
            "progress": project.progress,
            "health_score": project.health_score,
            "domain": project.domain,
            "start_date": project.start_date.isoformat(),
            "end_date": project.end_date.isoformat() if project.end_date else None
        }
    
    # Sauvegarde
    os.makedirs("data/demo", exist_ok=True)
    with open("data/demo/portfolio_soutenance.json", "w", encoding="utf-8") as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    
    print("[SAVE] [DEMO] Portfolio sauvegarde dans data/demo/portfolio_soutenance.json")

def print_demo_summary():
    """Affiche un résumé impressionnant pour la soutenance"""
    
    summary = get_demo_presentation_data()
    
    print("\n" + "="*60)
    print("PORTFOLIO EXEMPLAIRE - SOUTENANCE PLANNERAI")
    print("="*60)
    print(f"Total Projets: {summary['total_projects']}")
    print(f"Investissement Total: {summary['total_investment']:,.0f}EUR")
    print(f"Equipes Mobilisees: {summary['team_members']} personnes")
    print(f"Progression Moyenne: {summary['avg_progress']:.1f}%")
    print(f"ROI Projete: {summary['projected_roi']}")
    print(f"Impact Marche: {summary['market_impact']}")
    print(f"Utilisateurs Touches: {summary['expected_users']}")
    print("\nDOMAINES COUVERTS:")
    for domain in summary['domains_covered']:
        print(f"  - {domain}")
    print("\nINNOVATIONS TECHNOLOGIQUES:")
    for innovation in summary['innovation_areas']:
        print(f"  - {innovation}")
    print("="*60)
    print("[SUCCESS] Portfolio pret pour soutenance et video de presentation!")
    print("="*60)

if __name__ == "__main__":
    # Création du portfolio exemplaire
    projects = create_demo_portfolio()
    
    # Sauvegarde pour référence
    save_demo_portfolio()
    
    # Résumé pour présentation
    print_demo_summary()