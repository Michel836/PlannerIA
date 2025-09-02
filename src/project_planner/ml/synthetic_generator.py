"""
Générateur de données synthétiques avancé pour PlannerIA
Crée des tâches réalistes pour l'entraînement des modèles ML
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path


class EnhancedSyntheticGenerator:
    """Générateur de données synthétiques réalistes pour projets"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Configuration des domaines métiers
        self.business_domains = {
            'ecommerce': {
                'weight': 0.25,
                'complexity_bias': 1.2,
                'cost_multiplier': 1.1,
                'keywords': ['shop', 'cart', 'payment', 'product', 'catalog', 'checkout']
            },
            'fintech': {
                'weight': 0.15,
                'complexity_bias': 1.6,
                'cost_multiplier': 1.4,
                'keywords': ['payment', 'transaction', 'wallet', 'banking', 'compliance']
            },
            'healthcare': {
                'weight': 0.12,
                'complexity_bias': 1.7,
                'cost_multiplier': 1.5,
                'keywords': ['patient', 'medical', 'records', 'compliance', 'security']
            },
            'education': {
                'weight': 0.18,
                'complexity_bias': 1.0,
                'cost_multiplier': 0.9,
                'keywords': ['student', 'course', 'learning', 'assessment', 'content']
            },
            'enterprise': {
                'weight': 0.20,
                'complexity_bias': 1.8,
                'cost_multiplier': 1.6,
                'keywords': ['workflow', 'integration', 'dashboard', 'reporting', 'automation']
            },
            'startup': {
                'weight': 0.10,
                'complexity_bias': 0.8,
                'cost_multiplier': 0.7,
                'keywords': ['mvp', 'prototype', 'validation', 'launch', 'growth']
            }
        }
        
        # Types de tâches techniques détaillés
        self.task_categories = {
            'frontend': {
                'subcategories': {
                    'ui_design': {
                        'keywords': ['interface', 'wireframe', 'mockup', 'design system'],
                        'base_duration': 3.0,
                        'complexity_factor': 1.2
                    },
                    'component_dev': {
                        'keywords': ['component', 'react', 'vue', 'angular', 'widget'],
                        'base_duration': 4.0,
                        'complexity_factor': 1.3
                    },
                    'responsive': {
                        'keywords': ['responsive', 'mobile', 'tablet', 'adaptive'],
                        'base_duration': 3.5,
                        'complexity_factor': 1.1
                    },
                    'optimization': {
                        'keywords': ['performance', 'optimization', 'lazy loading', 'bundle'],
                        'base_duration': 5.0,
                        'complexity_factor': 1.6
                    }
                }
            },
            'backend': {
                'subcategories': {
                    'api_development': {
                        'keywords': ['api', 'endpoint', 'rest', 'graphql', 'microservice'],
                        'base_duration': 5.0,
                        'complexity_factor': 1.4
                    },
                    'database': {
                        'keywords': ['database', 'sql', 'migration', 'schema', 'indexing'],
                        'base_duration': 4.5,
                        'complexity_factor': 1.3
                    },
                    'integration': {
                        'keywords': ['integration', 'third-party', 'webhook', 'sync'],
                        'base_duration': 6.0,
                        'complexity_factor': 1.7
                    },
                    'security': {
                        'keywords': ['authentication', 'authorization', 'encryption', 'security'],
                        'base_duration': 7.0,
                        'complexity_factor': 1.9
                    }
                }
            },
            'devops': {
                'subcategories': {
                    'deployment': {
                        'keywords': ['deployment', 'ci/cd', 'pipeline', 'automation'],
                        'base_duration': 3.0,
                        'complexity_factor': 1.4
                    },
                    'infrastructure': {
                        'keywords': ['infrastructure', 'docker', 'kubernetes', 'cloud'],
                        'base_duration': 4.0,
                        'complexity_factor': 1.6
                    },
                    'monitoring': {
                        'keywords': ['monitoring', 'logging', 'metrics', 'alerting'],
                        'base_duration': 3.5,
                        'complexity_factor': 1.2
                    }
                }
            },
            'testing': {
                'subcategories': {
                    'unit_testing': {
                        'keywords': ['unit test', 'testing', 'coverage', 'jest'],
                        'base_duration': 2.5,
                        'complexity_factor': 1.0
                    },
                    'integration_testing': {
                        'keywords': ['integration', 'e2e', 'selenium', 'automation'],
                        'base_duration': 4.0,
                        'complexity_factor': 1.3
                    },
                    'performance_testing': {
                        'keywords': ['performance', 'load testing', 'stress', 'benchmark'],
                        'base_duration': 3.5,
                        'complexity_factor': 1.5
                    }
                }
            },
            'data': {
                'subcategories': {
                    'data_processing': {
                        'keywords': ['etl', 'data processing', 'pipeline', 'transformation'],
                        'base_duration': 5.5,
                        'complexity_factor': 1.4
                    },
                    'analytics': {
                        'keywords': ['analytics', 'reporting', 'dashboard', 'kpi'],
                        'base_duration': 4.0,
                        'complexity_factor': 1.2
                    },
                    'ml_ai': {
                        'keywords': ['machine learning', 'ai', 'model', 'prediction'],
                        'base_duration': 8.0,
                        'complexity_factor': 2.0
                    }
                }
            }
        }
        
        # Niveaux de complexité raffinés
        self.complexity_profiles = {
            'trivial': {
                'weight': 0.05,
                'duration_multiplier': 0.3,
                'cost_multiplier': 0.4,
                'risk_factor': 0.1,
                'descriptors': ['trivial', 'simple', 'quick fix', 'minor']
            },
            'simple': {
                'weight': 0.20,
                'duration_multiplier': 0.6,
                'cost_multiplier': 0.7,
                'risk_factor': 0.2,
                'descriptors': ['simple', 'basic', 'straightforward', 'standard']
            },
            'medium': {
                'weight': 0.40,
                'duration_multiplier': 1.0,
                'cost_multiplier': 1.0,
                'risk_factor': 0.3,
                'descriptors': ['standard', 'regular', 'typical', 'moderate']
            },
            'complex': {
                'weight': 0.25,
                'duration_multiplier': 1.6,
                'cost_multiplier': 1.5,
                'risk_factor': 0.5,
                'descriptors': ['complex', 'advanced', 'sophisticated', 'challenging']
            },
            'very_complex': {
                'weight': 0.08,
                'duration_multiplier': 2.5,
                'cost_multiplier': 2.2,
                'risk_factor': 0.7,
                'descriptors': ['highly complex', 'enterprise-grade', 'mission-critical']
            },
            'extreme': {
                'weight': 0.02,
                'duration_multiplier': 4.0,
                'cost_multiplier': 3.5,
                'risk_factor': 0.9,
                'descriptors': ['extremely complex', 'cutting-edge', 'research-level']
            }
        }
        
        # Templates de noms de tâches
        self.task_name_templates = [
            "{action} {descriptor} {component} {domain_context}",
            "{action} {component} with {feature}",
            "{descriptor} {component} {action}",
            "{action} {domain_context} {component}",
            "Implement {descriptor} {component}",
            "Develop {component} for {domain_context}",
            "Create {descriptor} {feature} system",
            "Build {component} with {feature} support"
        ]
        
        # Actions de développement
        self.actions = [
            'implement', 'develop', 'create', 'build', 'design', 'configure',
            'integrate', 'optimize', 'refactor', 'migrate', 'deploy', 'setup',
            'enhance', 'extend', 'customize', 'modernize'
        ]
        
        # Priorités avec distribution réaliste
        self.priority_distribution = {
            'low': 0.15,
            'medium': 0.50,
            'high': 0.25,
            'critical': 0.08,
            'urgent': 0.02
        }
    
    def generate_realistic_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Génère des tâches réalistes avec distribution cohérente"""
        tasks = []
        
        for i in range(num_tasks):
            # Sélection du domaine métier
            domain = self._select_weighted_choice(self.business_domains)
            domain_info = self.business_domains[domain]
            
            # Sélection de la catégorie technique
            category = random.choice(list(self.task_categories.keys()))
            subcategory = self._select_weighted_choice(
                self.task_categories[category]['subcategories']
            )
            
            # Sélection de la complexité
            complexity = self._select_weighted_choice(self.complexity_profiles)
            complexity_info = self.complexity_profiles[complexity]
            
            # Génération de la tâche
            task = self._generate_single_task(
                i, domain, domain_info, category, subcategory, 
                complexity, complexity_info
            )
            
            tasks.append(task)
        
        return tasks
    
    def _select_weighted_choice(self, choices_dict: Dict[str, Dict]) -> str:
        """Sélection pondérée d'un choix"""
        if 'weight' in list(choices_dict.values())[0]:
            weights = [info['weight'] for info in choices_dict.values()]
            choices = list(choices_dict.keys())
            return np.random.choice(choices, p=weights)
        else:
            return random.choice(list(choices_dict.keys()))
    
    def _generate_single_task(
        self, 
        index: int,
        domain: str, 
        domain_info: Dict[str, Any],
        category: str,
        subcategory: str,
        complexity: str,
        complexity_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère une tâche individuelle"""
        
        # Informations de base
        subcategory_info = self.task_categories[category]['subcategories'][subcategory]
        
        # Génération du nom et description
        task_name = self._generate_task_name(
            domain_info, subcategory_info, complexity_info
        )
        
        task_description = self._generate_task_description(
            domain, category, subcategory, complexity, task_name
        )
        
        # Calcul des estimations
        duration, cost = self._calculate_realistic_estimates(
            subcategory_info, complexity_info, domain_info
        )
        
        # Sélection de la priorité
        priority = self._select_weighted_choice(
            {k: {'weight': v} for k, v in self.priority_distribution.items()}
        )
        
        # Génération des équipes et ressources
        team_composition = self._generate_team_composition(complexity, category)
        
        # Dépendances et livrables
        dependencies = self._generate_dependencies(complexity, category)
        deliverables = self._generate_deliverables(category, subcategory, complexity)
        
        return {
            'id': f'synthetic_task_{index:05d}',
            'name': task_name,
            'description': task_description,
            'duration': duration,
            'cost': cost,
            'priority': priority,
            'task_type': category,
            'subcategory': subcategory,
            'complexity_level': complexity,
            'domain': domain,
            'team_size': len(team_composition['resources']),
            'assigned_resources': team_composition['resources'],
            'required_skills': team_composition['skills'],
            'dependencies': dependencies,
            'deliverables': deliverables,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'is_synthetic': True,
                'complexity_score': complexity_info['risk_factor'],
                'domain_multiplier': domain_info['complexity_bias']
            }
        }
    
    def _generate_task_name(
        self, 
        domain_info: Dict[str, Any], 
        subcategory_info: Dict[str, Any],
        complexity_info: Dict[str, Any]
    ) -> str:
        """Génère un nom de tâche réaliste"""
        
        template = random.choice(self.task_name_templates)
        
        components = {
            'action': random.choice(self.actions).title(),
            'descriptor': random.choice(complexity_info['descriptors']),
            'component': random.choice(subcategory_info['keywords']),
            'feature': random.choice([
                'authentication', 'validation', 'optimization', 'integration',
                'monitoring', 'caching', 'security', 'analytics'
            ]),
            'domain_context': random.choice(domain_info['keywords'])
        }
        
        # Sélection aléatoire des composants à utiliser
        used_components = {k: v for k, v in components.items() if random.random() > 0.3}
        
        try:
            name = template.format(**used_components)
            # Nettoyage des espaces multiples et formatage
            name = ' '.join(name.split())
            return name.title()
        except KeyError:
            # Fallback si le template ne peut pas être formaté
            return f"{components['action']} {components['descriptor']} {components['component']}"
    
    def _generate_task_description(
        self,
        domain: str,
        category: str, 
        subcategory: str,
        complexity: str,
        task_name: str
    ) -> str:
        """Génère une description détaillée de la tâche"""
        
        base_descriptions = {
            'frontend': [
                "Develop responsive user interface components with modern frameworks",
                "Create interactive user experience following design system guidelines",
                "Implement client-side functionality with performance optimization"
            ],
            'backend': [
                "Build scalable server-side architecture with proper error handling",
                "Develop robust API endpoints with comprehensive validation",
                "Implement data processing logic with security best practices"
            ],
            'devops': [
                "Configure automated deployment pipeline with monitoring",
                "Set up infrastructure as code with scalability considerations",
                "Implement CI/CD workflows with quality gates"
            ],
            'testing': [
                "Design comprehensive testing strategy covering edge cases",
                "Implement automated testing suite with coverage reporting",
                "Develop quality assurance processes and documentation"
            ],
            'data': [
                "Build data processing pipeline with validation and monitoring",
                "Implement analytics solution with real-time capabilities",
                "Design machine learning model with performance tracking"
            ]
        }
        
        base_desc = random.choice(base_descriptions.get(category, base_descriptions['backend']))
        
        # Ajout de spécificités selon la complexité
        complexity_additions = {
            'simple': ["with standard practices", "following established patterns"],
            'medium': ["with proper error handling", "including comprehensive testing"],
            'complex': ["with advanced features", "ensuring high performance and scalability"],
            'very_complex': ["with enterprise-grade requirements", "meeting strict compliance standards"],
            'extreme': ["with cutting-edge technology", "pushing performance boundaries"]
        }
        
        addition = random.choice(complexity_additions.get(complexity, complexity_additions['medium']))
        
        # Ajout de contexte métier
        domain_context = {
            'ecommerce': "for online retail platform",
            'fintech': "for financial services application", 
            'healthcare': "for medical records system",
            'education': "for learning management platform",
            'enterprise': "for corporate workflow system",
            'startup': "for MVP development"
        }
        
        context = domain_context.get(domain, "for business application")
        
        return f"{base_desc} {addition} {context}. Includes proper documentation, testing, and deployment considerations."
    
    def _calculate_realistic_estimates(
        self,
        subcategory_info: Dict[str, Any],
        complexity_info: Dict[str, Any], 
        domain_info: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calcule des estimations réalistes de durée et coût"""
        
        # Durée de base
        base_duration = subcategory_info['base_duration']
        
        # Application des multiplicateurs
        duration_multiplier = (
            complexity_info['duration_multiplier'] * 
            subcategory_info['complexity_factor'] * 
            domain_info['complexity_bias']
        )
        
        # Ajout de variance réaliste (distribution log-normale)
        variance_factor = np.random.lognormal(0, 0.2)
        
        final_duration = base_duration * duration_multiplier * variance_factor
        
        # Arrondi réaliste
        if final_duration < 1:
            final_duration = round(final_duration, 1)
        else:
            final_duration = round(final_duration, 0)
        
        # Calcul du coût
        # Taux journalier variable selon complexité et domaine
        base_daily_rate = 600  # Taux de base
        
        complexity_rate_multiplier = {
            'trivial': 0.7,
            'simple': 0.8,
            'medium': 1.0,
            'complex': 1.3,
            'very_complex': 1.6,
            'extreme': 2.0
        }
        
        daily_rate = base_daily_rate * complexity_rate_multiplier.get(
            list(complexity_info.keys())[0] if isinstance(complexity_info, dict) else 'medium',
            1.0
        )
        
        # Coût = durée × taux journalier × multiplicateur domaine × équipe
        team_size_factor = random.uniform(1.0, 3.0)  # Taille équipe variable
        
        base_cost = final_duration * daily_rate * team_size_factor * domain_info['cost_multiplier']
        
        # Variance coût (plus conservatrice que durée)
        cost_variance = np.random.lognormal(0, 0.15)
        final_cost = base_cost * cost_variance
        
        return max(0.5, final_duration), max(500.0, round(final_cost, 0))
    
    def _generate_team_composition(self, complexity: str, category: str) -> Dict[str, Any]:
        """Génère une composition d'équipe réaliste"""
        
        # Taille équipe basée sur complexité
        team_size_ranges = {
            'trivial': (1, 1),
            'simple': (1, 2),
            'medium': (2, 3),
            'complex': (3, 5),
            'very_complex': (4, 6),
            'extreme': (5, 8)
        }
        
        min_size, max_size = team_size_ranges.get(complexity, (2, 3))
        team_size = random.randint(min_size, max_size)
        
        # Rôles par catégorie
        role_pools = {
            'frontend': ['Frontend Developer', 'UI/UX Designer', 'Frontend Architect'],
            'backend': ['Backend Developer', 'DevOps Engineer', 'Database Specialist', 'Solutions Architect'],
            'devops': ['DevOps Engineer', 'Infrastructure Specialist', 'Security Engineer'],
            'testing': ['QA Engineer', 'Test Automation Specialist', 'Quality Analyst'],
            'data': ['Data Engineer', 'Data Scientist', 'ML Engineer', 'Analytics Specialist']
        }
        
        available_roles = role_pools.get(category, role_pools['backend'])
        
        # Sélection des ressources
        resources = []
        skills = set()
        
        for i in range(team_size):
            level = random.choice(['Junior', 'Mid-level', 'Senior', 'Lead'])
            role = random.choice(available_roles)
            
            resource_name = f"{level} {role} {i+1}"
            resources.append(resource_name)
            
            # Compétences associées
            category_skills = {
                'frontend': ['JavaScript', 'React', 'CSS', 'HTML', 'TypeScript'],
                'backend': ['Python', 'Java', 'SQL', 'API Design', 'Microservices'],
                'devops': ['Docker', 'Kubernetes', 'AWS', 'CI/CD', 'Terraform'],
                'testing': ['Selenium', 'Jest', 'Pytest', 'Test Planning', 'Automation'],
                'data': ['Python', 'SQL', 'Machine Learning', 'ETL', 'Analytics']
            }
            
            role_skills = category_skills.get(category, category_skills['backend'])
            selected_skills = random.sample(role_skills, random.randint(2, 4))
            skills.update(selected_skills)
        
        return {
            'resources': resources,
            'skills': list(skills)
        }
    
    def _generate_dependencies(self, complexity: str, category: str) -> List[str]:
        """Génère des dépendances réalistes"""
        
        dependency_counts = {
            'trivial': (0, 1),
            'simple': (0, 2),
            'medium': (1, 3),
            'complex': (2, 5),
            'very_complex': (3, 7),
            'extreme': (4, 10)
        }
        
        min_deps, max_deps = dependency_counts.get(complexity, (1, 3))
        num_dependencies = random.randint(min_deps, max_deps)
        
        if num_dependencies == 0:
            return []
        
        dependency_templates = [
            "Database schema setup",
            "Authentication system",
            "Third-party API integration",
            "Infrastructure provisioning",
            "Security audit completion",
            "Design system approval",
            "Performance baseline establishment",
            "Compliance review",
            "User acceptance testing",
            "Code review completion"
        ]
        
        return random.sample(dependency_templates, min(num_dependencies, len(dependency_templates)))
    
    def _generate_deliverables(self, category: str, subcategory: str, complexity: str) -> List[str]:
        """Génère des livrables réalistes"""
        
        base_deliverables = {
            'frontend': [
                'UI Components', 'Design Mockups', 'Responsive Layout',
                'User Interface Documentation', 'Accessibility Report'
            ],
            'backend': [
                'API Documentation', 'Database Schema', 'Service Implementation',
                'Integration Tests', 'Performance Benchmarks'
            ],
            'devops': [
                'Deployment Scripts', 'Infrastructure as Code', 'Monitoring Setup',
                'CI/CD Pipeline', 'Security Configuration'
            ],
            'testing': [
                'Test Cases', 'Automated Test Suite', 'Test Reports',
                'Quality Metrics', 'Bug Reports'
            ],
            'data': [
                'Data Pipeline', 'Analytics Dashboard', 'Data Models',
                'Performance Metrics', 'Documentation'
            ]
        }
        
        category_deliverables = base_deliverables.get(category, base_deliverables['backend'])
        
        # Nombre de livrables basé sur complexité
        deliverable_counts = {
            'trivial': (1, 2),
            'simple': (2, 3),
            'medium': (3, 4),
            'complex': (4, 6),
            'very_complex': (5, 7),
            'extreme': (6, 8)
        }
        
        min_count, max_count = deliverable_counts.get(complexity, (3, 4))
        num_deliverables = random.randint(min_count, max_count)
        
        selected_deliverables = random.sample(
            category_deliverables, 
            min(num_deliverables, len(category_deliverables))
        )
        
        # Ajout de livrables communs
        common_deliverables = [
            'Technical Documentation', 'Code Review Report', 
            'Unit Tests', 'Deployment Guide'
        ]
        
        # Ajout aléatoire de livrables communs
        for common_del in common_deliverables:
            if random.random() > 0.6 and common_del not in selected_deliverables:
                selected_deliverables.append(common_del)
        
        return selected_deliverables
    
    def save_generated_data(self, tasks: List[Dict[str, Any]], filename: str = None) -> Path:
        """Sauvegarde les données générées"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_data_{timestamp}.json"
        
        output_dir = Path("data/training")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        
        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_tasks': len(tasks),
                'generator_version': '2.0',
                'complexity_distribution': self._analyze_complexity_distribution(tasks),
                'category_distribution': self._analyze_category_distribution(tasks)
            },
            'tasks': tasks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path
    
    def _analyze_complexity_distribution(self, tasks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyse la distribution de complexité"""
        distribution = {}
        for task in tasks:
            complexity = task.get('complexity_level', 'unknown')
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def _analyze_category_distribution(self, tasks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyse la distribution par catégorie"""
        distribution = {}
        for task in tasks:
            category = task.get('task_type', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def generate_project_context(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Génère un contexte de projet cohérent pour les tâches"""
        
        # Analyse des tâches pour déduire le contexte
        domains = [task.get('domain', 'enterprise') for task in tasks]
        most_common_domain = max(set(domains), key=domains.count)
        
        total_duration = sum(task.get('duration', 0) for task in tasks)
        total_cost = sum(task.get('cost', 0) for task in tasks)
        
        project_names = {
            'ecommerce': ['ShopFlow', 'MarketPlace Pro', 'E-Commerce Suite'],
            'fintech': ['PaySecure', 'FinTech Platform', 'Digital Wallet'],
            'healthcare': ['MedRecords', 'HealthTech System', 'Patient Portal'],
            'education': ['EduPlatform', 'Learning Hub', 'Academic Suite'],
            'enterprise': ['Enterprise Suite', 'Business Platform', 'Corporate System'],
            'startup': ['MVP Platform', 'Startup Solution', 'Innovation Hub']
        }
        
        project_name = random.choice(project_names.get(most_common_domain, project_names['enterprise']))
        
        return {
            'project_name': project_name,
            'domain': most_common_domain,
            'total_tasks': len(tasks),
            'estimated_duration': total_duration,
            'estimated_cost': total_cost,
            'complexity_score': np.mean([task.get('metadata', {}).get('complexity_score', 0.5) for task in tasks]),
            'team_size_range': [
                min(task.get('team_size', 1) for task in tasks),
                max(task.get('team_size', 1) for task in tasks)
            ]
        }