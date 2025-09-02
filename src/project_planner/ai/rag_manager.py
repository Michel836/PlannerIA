"""
🔍 RAG Manager Intelligent - PlannerIA
Gestionnaire RAG adaptatif avec auto-enrichissement et intelligence contextuelle
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import os
import requests
from pathlib import Path
import hashlib
import re
from collections import defaultdict, Counter
import pickle

# RAG et ML imports
try:
    import faiss
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Web scraping
try:
    import beautifulsoup4
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False


class DocumentType(Enum):
    """Types de documents RAG"""
    INTERNAL_DOC = "internal_doc"
    WEB_ARTICLE = "web_article"
    CASE_STUDY = "case_study"
    BEST_PRACTICE = "best_practice"
    METHODOLOGY = "methodology"
    INDUSTRY_REPORT = "industry_report"
    ACADEMIC_PAPER = "academic_paper"
    TEMPLATE = "template"


class SourceReliability(Enum):
    """Niveaux de fiabilité des sources"""
    VERY_HIGH = "very_high"  # Sources académiques, grandes entreprises
    HIGH = "high"           # Sources reconnues, experts
    MEDIUM = "medium"       # Blogs techniques, forums spécialisés
    LOW = "low"            # Sources non vérifiées
    UNKNOWN = "unknown"     # Non évalué


class QueryContext(Enum):
    """Contextes de requête RAG"""
    PLANNING = "planning"
    BUDGET = "budget"
    RISK_MANAGEMENT = "risk_management" 
    CRISIS_HANDLING = "crisis_handling"
    TEAM_LEADERSHIP = "team_leadership"
    NEGOTIATION = "negotiation"
    MARKET_ANALYSIS = "market_analysis"
    TECHNOLOGY_TRENDS = "technology_trends"
    GENERAL = "general"


@dataclass
class DocumentMetadata:
    """Métadonnées enrichies d'un document"""
    id: str
    title: str
    source_url: Optional[str] = None
    document_type: DocumentType = DocumentType.INTERNAL_DOC
    reliability: SourceReliability = SourceReliability.UNKNOWN
    language: str = "fr"
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    topics: List[str] = field(default_factory=list)
    quality_score: float = 0.5  # 0-1
    usage_count: int = 0
    relevance_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RAGChunk:
    """Chunk de document avec métadonnées"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Résultat d'une requête RAG"""
    query: str
    context: QueryContext
    chunks: List[RAGChunk]
    confidence_score: float
    sources: List[DocumentMetadata]
    generated_answer: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class EnrichmentRecommendation:
    """Recommandation d'enrichissement"""
    topic: str
    priority: str  # "high", "medium", "low"
    suggested_sources: List[str]
    gap_description: str
    potential_impact: float  # 0-1
    search_queries: List[str]


class RAGManagerIntelligent:
    """Gestionnaire RAG intelligent et adaptatif"""
    
    def __init__(self, data_dir: str = "data/rag"):
        self.data_dir = data_dir
        self.embeddings_dir = os.path.join(data_dir, "embeddings")
        self.documents_dir = os.path.join(data_dir, "documents") 
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        self.analytics_file = os.path.join(data_dir, "analytics.json")
        
        # Initialisation des répertoires
        for directory in [self.data_dir, self.embeddings_dir, self.documents_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # État du système
        self.documents: Dict[str, DocumentMetadata] = {}
        self.chunks: Dict[str, RAGChunk] = {}
        self.analytics = {
            'total_queries': 0,
            'top_queries': Counter(),
            'context_usage': Counter(),
            'document_usage': Counter(),
            'last_enrichment': None,
            'enrichment_queue': []
        }
        
        # Modèles et index
        self.embedding_model = None
        self.faiss_index = None
        self.chunk_ids: List[str] = []
        
        # Configuration
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.max_results = 10
        self.confidence_threshold = 0.6
        
        # Sources de confiance pour auto-enrichissement
        self.trusted_sources = {
            'high_quality': [
                'harvard.edu', 'mit.edu', 'stanford.edu',
                'mckinsey.com', 'bcg.com', 'deloitte.com',
                'gartner.com', 'forrester.com',
                'projectmanagement.com', 'pmi.org'
            ],
            'medium_quality': [
                'medium.com', 'towardsdatascience.com',
                'stackoverflow.com', 'github.com',
                'atlassian.com', 'asana.com', 'monday.com'
            ]
        }
        
        # Patterns de qualité
        self.quality_patterns = {
            'high_quality_indicators': [
                r'\b(research|study|analysis|methodology)\b',
                r'\b(Harvard|MIT|Stanford|McKinsey|BCG)\b',
                r'\b(peer.reviewed|academic|journal)\b'
            ],
            'low_quality_indicators': [
                r'\b(click here|buy now|affiliate)\b',
                r'\b(get rich quick|amazing secret)\b'
            ]
        }
        
        # Initialisation
        self._load_system_state()
        self._initialize_embedding_model()
    
    def _load_system_state(self):
        """Charge l'état du système RAG"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                    
                # Reconstruction des objets DocumentMetadata
                for doc_id, doc_data in metadata_data.get('documents', {}).items():
                    doc_data['document_type'] = DocumentType(doc_data.get('document_type', 'internal_doc'))
                    doc_data['reliability'] = SourceReliability(doc_data.get('reliability', 'unknown'))
                    if doc_data.get('publication_date'):
                        doc_data['publication_date'] = datetime.fromisoformat(doc_data['publication_date'])
                    if doc_data.get('last_updated'):
                        doc_data['last_updated'] = datetime.fromisoformat(doc_data['last_updated'])
                    
                    self.documents[doc_id] = DocumentMetadata(id=doc_id, **doc_data)
            
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    analytics_data = json.load(f)
                    self.analytics.update(analytics_data)
                    
                    # Reconstruction des Counter objects
                    for key in ['top_queries', 'context_usage', 'document_usage']:
                        if key in self.analytics:
                            self.analytics[key] = Counter(self.analytics[key])
                            
        except Exception as e:
            print(f"Erreur chargement état RAG: {e}")
    
    def _save_system_state(self):
        """Sauvegarde l'état du système RAG"""
        try:
            # Préparation métadonnées pour sérialisation
            serializable_docs = {}
            for doc_id, doc in self.documents.items():
                doc_dict = doc.__dict__.copy()
                doc_dict['document_type'] = doc.document_type.value
                doc_dict['reliability'] = doc.reliability.value
                if doc_dict.get('publication_date'):
                    doc_dict['publication_date'] = doc_dict['publication_date'].isoformat()
                if doc_dict.get('last_updated'):
                    doc_dict['last_updated'] = doc_dict['last_updated'].isoformat()
                serializable_docs[doc_id] = doc_dict
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump({'documents': serializable_docs}, f, indent=2, ensure_ascii=False)
            
            # Sauvegarde analytics
            serializable_analytics = self.analytics.copy()
            for key in ['top_queries', 'context_usage', 'document_usage']:
                if key in serializable_analytics:
                    serializable_analytics[key] = dict(serializable_analytics[key])
            
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_analytics, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Erreur sauvegarde état RAG: {e}")
    
    def _initialize_embedding_model(self):
        """Initialise le modèle d'embeddings"""
        if not EMBEDDINGS_AVAILABLE:
            print("Embeddings non disponibles. Installez sentence-transformers et faiss-cpu")
            return
        
        try:
            # Modèle multilingue optimisé
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Chargement ou création de l'index FAISS
            index_file = os.path.join(self.embeddings_dir, "faiss_index.bin")
            chunk_ids_file = os.path.join(self.embeddings_dir, "chunk_ids.pkl")
            
            if os.path.exists(index_file) and os.path.exists(chunk_ids_file):
                self.faiss_index = faiss.read_index(index_file)
                with open(chunk_ids_file, 'rb') as f:
                    self.chunk_ids = pickle.load(f)
                print(f"Index FAISS chargé: {len(self.chunk_ids)} chunks")
            else:
                # Création d'un index vide
                dimension = 384  # Dimension du modèle MiniLM
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product pour similarité
                self.chunk_ids = []
                print("Nouvel index FAISS créé")
                
        except Exception as e:
            print(f"Erreur initialisation embeddings: {e}")
            self.embedding_model = None
    
    def add_document(self, content: str, metadata: DocumentMetadata) -> bool:
        """Ajoute un document au système RAG"""
        try:
            # Génération d'ID unique si non fourni
            if not metadata.id:
                content_hash = hashlib.md5(content.encode()).hexdigest()
                metadata.id = f"doc_{content_hash[:12]}"
            
            # Chunking intelligent
            chunks = self._intelligent_chunking(content, metadata)
            
            # Génération d'embeddings
            if self.embedding_model:
                embeddings = []
                chunk_texts = [chunk.content for chunk in chunks]
                
                if chunk_texts:
                    embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
                    
                    # Ajout à l'index FAISS
                    if embeddings.size > 0:
                        # Normalisation pour similarité cosinus
                        faiss.normalize_L2(embeddings)
                        
                        # Ajout à l'index
                        self.faiss_index.add(embeddings)
                        
                        # Mise à jour des mappings
                        for i, chunk in enumerate(chunks):
                            chunk.embedding = embeddings[i]
                            self.chunks[chunk.id] = chunk
                            self.chunk_ids.append(chunk.id)
            
            # Analyse de qualité
            metadata.quality_score = self._evaluate_document_quality(content, metadata)
            metadata.word_count = len(content.split())
            metadata.topics = self._extract_topics(content)
            
            # Sauvegarde du document
            doc_file = os.path.join(self.documents_dir, f"{metadata.id}.txt")
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Enregistrement métadonnées
            self.documents[metadata.id] = metadata
            
            # Sauvegarde de l'index
            self._save_faiss_index()
            self._save_system_state()
            
            print(f"Document ajouté: {metadata.title} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"Erreur ajout document: {e}")
            return False
    
    def _intelligent_chunking(self, content: str, metadata: DocumentMetadata) -> List[RAGChunk]:
        """Chunking intelligent basé sur le type de document"""
        chunks = []
        
        # Chunking adaptatif selon le type
        if metadata.document_type == DocumentType.ACADEMIC_PAPER:
            # Chunking par sections pour les papiers académiques
            sections = self._split_by_sections(content)
        elif metadata.document_type == DocumentType.CASE_STUDY:
            # Chunking par problème/solution pour les cas d'études
            sections = self._split_by_case_structure(content)
        else:
            # Chunking standard par taille
            sections = self._split_by_size(content)
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:  # Ignorer les chunks trop petits
                chunk_id = f"{metadata.id}_chunk_{i}"
                chunk = RAGChunk(
                    id=chunk_id,
                    document_id=metadata.id,
                    content=section.strip(),
                    chunk_index=i,
                    metadata={
                        'document_type': metadata.document_type.value,
                        'reliability': metadata.reliability.value
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Découpage par sections (titres, sous-titres)"""
        # Pattern pour détecter les titres
        section_pattern = r'(?m)^#{1,3}\s+.+$|^[A-Z][^.!?]*[.!?]\s*$'
        sections = re.split(section_pattern, content)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_by_case_structure(self, content: str) -> List[str]:
        """Découpage par structure de cas (contexte, problème, solution)"""
        # Patterns typiques des cas d'études
        structure_patterns = [
            r'(?i)(contexte|context|background)',
            r'(?i)(problème|problem|challenge|issue)',
            r'(?i)(solution|approach|methodology)',
            r'(?i)(résultat|result|outcome|conclusion)'
        ]
        
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            if any(re.search(pattern, line) for pattern in structure_patterns):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections if sections else self._split_by_size(content)
    
    def _split_by_size(self, content: str) -> List[str]:
        """Découpage standard par taille avec overlap"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def _evaluate_document_quality(self, content: str, metadata: DocumentMetadata) -> float:
        """Évalue la qualité d'un document"""
        score = 0.5  # Score de base
        
        # Bonus basé sur la source
        if metadata.source_url:
            domain = self._extract_domain(metadata.source_url)
            if domain in self.trusted_sources['high_quality']:
                score += 0.3
            elif domain in self.trusted_sources['medium_quality']:
                score += 0.1
        
        # Bonus basé sur les patterns de qualité
        for pattern in self.quality_patterns['high_quality_indicators']:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.1
        
        # Malus basé sur les patterns de faible qualité
        for pattern in self.quality_patterns['low_quality_indicators']:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 0.2
        
        # Bonus longueur (documents plus longs souvent plus détaillés)
        word_count = len(content.split())
        if word_count > 1000:
            score += 0.1
        elif word_count > 2000:
            score += 0.2
        
        # Bonus metadata complètes
        if metadata.author:
            score += 0.05
        if metadata.publication_date:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extraction de topics basique"""
        # Mots-clés de gestion de projet
        project_keywords = {
            'planning': ['planning', 'planification', 'schedule', 'timeline', 'gantt'],
            'budget': ['budget', 'cost', 'coût', 'financial', 'pricing'],
            'risk': ['risk', 'risque', 'threat', 'menace', 'uncertainty'],
            'team': ['team', 'équipe', 'leadership', 'management', 'collaboration'],
            'agile': ['agile', 'scrum', 'sprint', 'kanban', 'methodology'],
            'quality': ['quality', 'qualité', 'testing', 'QA', 'validation']
        }
        
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in project_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _extract_domain(self, url: str) -> str:
        """Extrait le domaine d'une URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return ""
    
    def query(self, question: str, context: QueryContext = QueryContext.GENERAL, 
              max_results: int = None, filters: Dict[str, Any] = None) -> RAGResult:
        """Requête RAG intelligente avec contexte"""
        start_time = datetime.now()
        
        if max_results is None:
            max_results = self.max_results
        
        # Analytics
        self.analytics['total_queries'] += 1
        self.analytics['top_queries'][question] += 1
        self.analytics['context_usage'][context.value] += 1
        
        try:
            if not self.embedding_model or not self.faiss_index:
                return self._fallback_search(question, context, max_results)
            
            # Génération embedding de la requête
            query_embedding = self.embedding_model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Recherche dans l'index FAISS
            k = min(max_results * 2, len(self.chunk_ids))  # Chercher plus pour filtrer
            if k == 0:
                return RAGResult(
                    query=question,
                    context=context,
                    chunks=[],
                    confidence_score=0.0,
                    sources=[],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Filtrage et ranking des résultats
            relevant_chunks = []
            seen_docs = set()
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= len(self.chunk_ids):
                    continue
                    
                chunk_id = self.chunk_ids[idx]
                if chunk_id not in self.chunks:
                    continue
                
                chunk = self.chunks[chunk_id]
                
                # Application des filtres
                if filters and not self._apply_filters(chunk, filters):
                    continue
                
                # Filtrage par confiance
                if score < self.confidence_threshold:
                    continue
                
                # Boost contextuel
                contextual_score = self._calculate_contextual_relevance(chunk, context)
                final_score = score * 0.7 + contextual_score * 0.3
                
                chunk_copy = RAGChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    metadata={**chunk.metadata, 'relevance_score': final_score}
                )
                
                relevant_chunks.append(chunk_copy)
                seen_docs.add(chunk.document_id)
            
            # Tri par score final
            relevant_chunks.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
            relevant_chunks = relevant_chunks[:max_results]
            
            # Récupération des métadonnées sources
            sources = [self.documents[doc_id] for doc_id in seen_docs if doc_id in self.documents]
            
            # Mise à jour usage
            for doc_id in seen_docs:
                if doc_id in self.documents:
                    self.documents[doc_id].usage_count += 1
                    self.analytics['document_usage'][doc_id] += 1
            
            # Calcul de confiance globale
            if relevant_chunks:
                confidence = np.mean([chunk.metadata['relevance_score'] for chunk in relevant_chunks])
            else:
                confidence = 0.0
            
            # Génération de réponse (optionnel)
            generated_answer = self._generate_answer(question, relevant_chunks, context)
            
            # Citations
            citations = self._generate_citations(relevant_chunks)
            
            result = RAGResult(
                query=question,
                context=context,
                chunks=relevant_chunks,
                confidence_score=confidence,
                sources=sources,
                generated_answer=generated_answer,
                citations=citations,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Sauvegarde périodique des analytics
            if self.analytics['total_queries'] % 10 == 0:
                self._save_system_state()
            
            return result
            
        except Exception as e:
            print(f"Erreur requête RAG: {e}")
            return RAGResult(
                query=question,
                context=context,
                chunks=[],
                confidence_score=0.0,
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _apply_filters(self, chunk: RAGChunk, filters: Dict[str, Any]) -> bool:
        """Applique les filtres à un chunk"""
        for filter_key, filter_value in filters.items():
            if filter_key == 'document_type':
                if chunk.metadata.get('document_type') != filter_value:
                    return False
            elif filter_key == 'reliability':
                if chunk.metadata.get('reliability') != filter_value:
                    return False
            elif filter_key == 'min_quality':
                doc_id = chunk.document_id
                if doc_id in self.documents:
                    if self.documents[doc_id].quality_score < filter_value:
                        return False
        
        return True
    
    def _calculate_contextual_relevance(self, chunk: RAGChunk, context: QueryContext) -> float:
        """Calcule la pertinence contextuelle d'un chunk"""
        base_score = 0.5
        
        # Mapping contexte -> mots-clés
        context_keywords = {
            QueryContext.PLANNING: ['planning', 'schedule', 'timeline', 'gantt', 'milestone'],
            QueryContext.BUDGET: ['budget', 'cost', 'financial', 'pricing', 'estimation'],
            QueryContext.RISK_MANAGEMENT: ['risk', 'threat', 'mitigation', 'contingency'],
            QueryContext.CRISIS_HANDLING: ['crisis', 'emergency', 'escalation', 'recovery'],
            QueryContext.TEAM_LEADERSHIP: ['team', 'leadership', 'management', 'motivation'],
            QueryContext.NEGOTIATION: ['negotiation', 'agreement', 'stakeholder', 'compromise'],
            QueryContext.MARKET_ANALYSIS: ['market', 'competitive', 'analysis', 'trends'],
            QueryContext.TECHNOLOGY_TRENDS: ['technology', 'innovation', 'digital', 'AI']
        }
        
        keywords = context_keywords.get(context, [])
        content_lower = chunk.content.lower()
        
        # Calcul de la pertinence basée sur les mots-clés
        keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
        if keywords:
            keyword_score = keyword_matches / len(keywords)
            base_score += keyword_score * 0.3
        
        # Bonus pour documents du bon type
        doc_type = chunk.metadata.get('document_type', '')
        type_bonuses = {
            QueryContext.PLANNING: ['methodology', 'template'],
            QueryContext.BUDGET: ['industry_report'],
            QueryContext.RISK_MANAGEMENT: ['case_study', 'best_practice'],
            QueryContext.CRISIS_HANDLING: ['case_study'],
            QueryContext.TEAM_LEADERSHIP: ['best_practice'],
            QueryContext.MARKET_ANALYSIS: ['industry_report'],
            QueryContext.TECHNOLOGY_TRENDS: ['industry_report', 'academic_paper']
        }
        
        if context in type_bonuses and doc_type in type_bonuses[context]:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _fallback_search(self, question: str, context: QueryContext, max_results: int) -> RAGResult:
        """Recherche de fallback sans embeddings"""
        # Recherche textuelle simple
        question_words = set(question.lower().split())
        matching_chunks = []
        
        for chunk in self.chunks.values():
            chunk_words = set(chunk.content.lower().split())
            intersection = question_words.intersection(chunk_words)
            
            if intersection:
                score = len(intersection) / len(question_words)
                if score > 0.2:  # Seuil minimum
                    chunk_copy = RAGChunk(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        metadata={**chunk.metadata, 'relevance_score': score}
                    )
                    matching_chunks.append(chunk_copy)
        
        # Tri par score
        matching_chunks.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
        matching_chunks = matching_chunks[:max_results]
        
        sources = []
        for chunk in matching_chunks:
            if chunk.document_id in self.documents:
                doc = self.documents[chunk.document_id]
                if doc not in sources:
                    sources.append(doc)
        
        confidence = np.mean([c.metadata['relevance_score'] for c in matching_chunks]) if matching_chunks else 0.0
        
        return RAGResult(
            query=question,
            context=context,
            chunks=matching_chunks,
            confidence_score=confidence,
            sources=sources
        )
    
    def _generate_answer(self, question: str, chunks: List[RAGChunk], context: QueryContext) -> str:
        """Génère une réponse basée sur les chunks (version simple)"""
        if not chunks:
            return "Aucune information pertinente trouvée dans la base de connaissances."
        
        # Concaténation des chunks les plus pertinents
        relevant_content = []
        for chunk in chunks[:3]:  # Top 3 chunks
            relevant_content.append(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
        
        combined_content = "\n\n".join(relevant_content)
        
        # Réponse simple basée sur le contexte
        context_intro = {
            QueryContext.PLANNING: "Concernant la planification de projet :",
            QueryContext.BUDGET: "Concernant la gestion budgétaire :",
            QueryContext.RISK_MANAGEMENT: "Concernant la gestion des risques :",
            QueryContext.CRISIS_HANDLING: "Concernant la gestion de crise :",
            QueryContext.TEAM_LEADERSHIP: "Concernant le leadership d'équipe :",
            QueryContext.NEGOTIATION: "Concernant la négociation :",
            QueryContext.MARKET_ANALYSIS: "Concernant l'analyse de marché :",
            QueryContext.TECHNOLOGY_TRENDS: "Concernant les tendances technologiques :",
        }
        
        intro = context_intro.get(context, "D'après la documentation disponible :")
        
        return f"{intro}\n\n{combined_content}"
    
    def _generate_citations(self, chunks: List[RAGChunk]) -> List[str]:
        """Génère des citations pour les sources"""
        citations = []
        seen_docs = set()
        
        for chunk in chunks[:5]:  # Top 5 sources
            if chunk.document_id not in seen_docs and chunk.document_id in self.documents:
                doc = self.documents[chunk.document_id]
                citation = f"{doc.title}"
                
                if doc.author:
                    citation += f" - {doc.author}"
                
                if doc.source_url:
                    citation += f" ({doc.source_url})"
                
                citations.append(citation)
                seen_docs.add(chunk.document_id)
        
        return citations
    
    def auto_enrich_from_web(self, topics: List[str], max_docs_per_topic: int = 3) -> List[EnrichmentRecommendation]:
        """Auto-enrichissement depuis le web"""
        recommendations = []
        
        if not WEB_SCRAPING_AVAILABLE:
            print("Web scraping non disponible. Installez beautifulsoup4")
            return recommendations
        
        for topic in topics:
            try:
                # Génération de requêtes de recherche
                search_queries = self._generate_search_queries(topic)
                suggested_sources = []
                
                # Simulation de recherche (en réalité, utiliserait une API de recherche)
                for domain in self.trusted_sources['high_quality'][:5]:
                    suggested_sources.append(f"https://{domain}/search?q={topic}")
                
                recommendation = EnrichmentRecommendation(
                    topic=topic,
                    priority="medium",
                    suggested_sources=suggested_sources,
                    gap_description=f"Manque de documentation sur {topic}",
                    potential_impact=0.7,
                    search_queries=search_queries
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                print(f"Erreur enrichissement pour {topic}: {e}")
        
        return recommendations
    
    def _generate_search_queries(self, topic: str) -> List[str]:
        """Génère des requêtes de recherche pour un topic"""
        base_queries = {
            'planning': [
                "project planning methodology best practices",
                "agile project planning techniques", 
                "project timeline estimation methods"
            ],
            'budget': [
                "project budget estimation techniques",
                "cost management best practices",
                "project financial planning methods"
            ],
            'risk': [
                "project risk management framework",
                "risk assessment methodologies",
                "project risk mitigation strategies"
            ],
            'team': [
                "team leadership project management",
                "project team collaboration techniques",
                "remote team management best practices"
            ]
        }
        
        return base_queries.get(topic, [f"{topic} project management best practices"])
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Retourne les analytics pour le dashboard"""
        total_docs = len(self.documents)
        total_chunks = len(self.chunks)
        
        # Calcul de métriques
        if self.documents:
            avg_quality = np.mean([doc.quality_score for doc in self.documents.values()])
            quality_distribution = Counter()
            for doc in self.documents.values():
                if doc.quality_score >= 0.8:
                    quality_distribution['high'] += 1
                elif doc.quality_score >= 0.5:
                    quality_distribution['medium'] += 1
                else:
                    quality_distribution['low'] += 1
        else:
            avg_quality = 0
            quality_distribution = Counter()
        
        # Top documents utilisés
        top_documents = []
        if self.documents:
            sorted_docs = sorted(self.documents.values(), key=lambda x: x.usage_count, reverse=True)
            for doc in sorted_docs[:10]:
                top_documents.append({
                    'title': doc.title,
                    'usage_count': doc.usage_count,
                    'quality_score': doc.quality_score,
                    'document_type': doc.document_type.value
                })
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'total_queries': self.analytics['total_queries'],
            'average_quality': avg_quality,
            'quality_distribution': dict(quality_distribution),
            'top_queries': dict(self.analytics['top_queries'].most_common(10)),
            'context_usage': dict(self.analytics['context_usage']),
            'top_documents': top_documents,
            'index_status': 'active' if self.faiss_index else 'inactive',
            'embedding_model': 'MiniLM-L12' if self.embedding_model else 'none',
            'last_enrichment': self.analytics.get('last_enrichment'),
            'enrichment_queue_size': len(self.analytics.get('enrichment_queue', []))
        }
    
    def suggest_missing_topics(self) -> List[EnrichmentRecommendation]:
        """Suggère des topics manquants basés sur l'usage"""
        # Analyse des gaps basés sur les requêtes sans résultats satisfaisants
        gap_topics = []
        
        # Analyse des contexts peu couverts
        context_coverage = {}
        for context in QueryContext:
            relevant_docs = 0
            for doc in self.documents.values():
                if context.value in [topic.lower() for topic in doc.topics]:
                    relevant_docs += 1
            context_coverage[context.value] = relevant_docs
        
        # Identification des gaps
        for context, doc_count in context_coverage.items():
            if doc_count < 3:  # Seuil arbitraire
                gap_topics.append(context)
        
        # Génération de recommandations
        recommendations = []
        for topic in gap_topics:
            rec = EnrichmentRecommendation(
                topic=topic,
                priority="high" if context_coverage[topic] == 0 else "medium",
                suggested_sources=self._generate_search_queries(topic),
                gap_description=f"Seulement {context_coverage[topic]} documents sur {topic}",
                potential_impact=0.8,
                search_queries=self._generate_search_queries(topic)
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _save_faiss_index(self):
        """Sauvegarde l'index FAISS"""
        if self.faiss_index:
            try:
                index_file = os.path.join(self.embeddings_dir, "faiss_index.bin")
                chunk_ids_file = os.path.join(self.embeddings_dir, "chunk_ids.pkl")
                
                faiss.write_index(self.faiss_index, index_file)
                
                with open(chunk_ids_file, 'wb') as f:
                    pickle.dump(self.chunk_ids, f)
                    
            except Exception as e:
                print(f"Erreur sauvegarde index FAISS: {e}")


# Instance globale du RAG Manager
ai_rag_manager = RAGManagerIntelligent()

# Fonctions utilitaires pour Streamlit
def initialize_rag_manager():
    """Initialise le RAG manager dans Streamlit"""
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = ai_rag_manager
    return st.session_state.rag_manager


def get_rag_manager() -> RAGManagerIntelligent:
    """Récupère l'instance du RAG manager"""
    return getattr(st.session_state, 'rag_manager', ai_rag_manager)