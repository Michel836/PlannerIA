"""
Enhanced Error Handling System for PlannerIA
Provides user-friendly error messages and recovery suggestions
"""

import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)

class PlannerIAError(Exception):
    """Base exception for PlannerIA specific errors"""
    def __init__(self, message: str, error_code: str = None, recovery_suggestions: list = None):
        super().__init__(message)
        self.error_code = error_code
        self.recovery_suggestions = recovery_suggestions or []

class LLMConnectionError(PlannerIAError):
    """Error when LLM connection fails"""
    pass

class ValidationError(PlannerIAError):
    """Error when input validation fails"""
    pass

class ProcessingError(PlannerIAError):
    """Error during plan processing"""
    pass

class ErrorHandler:
    """Enhanced error handling with user-friendly messages"""
    
    # Error categories with user-friendly messages
    ERROR_CATEGORIES = {
        "llm_connection": {
            "title": "🔌 Problème de Connexion LLM",
            "description": "Impossible de se connecter au modèle de langage",
            "suggestions": [
                "Vérifiez qu'Ollama est démarré (ollama serve)",
                "Confirmez que le modèle llama3.2:latest est installé",
                "Vérifiez la connectivité réseau locale",
                "Redémarrez le service Ollama si nécessaire"
            ],
            "technical_check": "Connexion Ollama sur http://localhost:11434"
        },
        "validation_error": {
            "title": "📝 Erreur de Validation",
            "description": "Les données fournies ne respectent pas le format attendu",
            "suggestions": [
                "Vérifiez que votre description de projet est complète",
                "Assurez-vous d'inclure les objectifs et le périmètre",
                "Utilisez un langage clair et structuré",
                "Évitez les caractères spéciaux dans les noms"
            ],
            "technical_check": "Validation des données d'entrée"
        },
        "processing_error": {
            "title": "⚙️ Erreur de Traitement",
            "description": "Une erreur s'est produite pendant la génération du plan",
            "suggestions": [
                "Simplifiez votre demande de projet",
                "Réduisez la complexité des exigences",
                "Tentez à nouveau dans quelques secondes",
                "Redémarrez l'application si le problème persiste"
            ],
            "technical_check": "Pipeline de traitement des agents IA"
        },
        "export_error": {
            "title": "📄 Erreur d'Export",
            "description": "Impossible de générer le fichier d'export",
            "suggestions": [
                "Vérifiez l'espace disque disponible",
                "Assurez-vous que le dossier de destination est accessible",
                "Fermez les fichiers ouverts du même nom",
                "Tentez un export dans un format différent"
            ],
            "technical_check": "Génération de rapports"
        },
        "system_error": {
            "title": "🖥️ Erreur Système",
            "description": "Une erreur système inattendue s'est produite",
            "suggestions": [
                "Redémarrez l'application",
                "Vérifiez les ressources système (RAM, CPU)",
                "Consultez les logs pour plus de détails",
                "Contactez l'support technique si le problème persiste"
            ],
            "technical_check": "Système et ressources"
        }
    }
    
    @staticmethod
    def classify_error(error: Exception) -> str:
        """Classify error type based on exception content"""
        error_str = str(error).lower()
        
        # LLM connection errors
        if any(keyword in error_str for keyword in ["connection", "ollama", "http", "timeout", "refused"]):
            return "llm_connection"
            
        # Validation errors
        if any(keyword in error_str for keyword in ["validation", "invalid", "missing", "required", "schema"]):
            return "validation_error"
            
        # Export errors
        if any(keyword in error_str for keyword in ["pdf", "csv", "export", "file", "write", "permission"]):
            return "export_error"
            
        # Processing errors
        if any(keyword in error_str for keyword in ["processing", "generation", "agent", "crew", "pipeline"]):
            return "processing_error"
            
        # Default to system error
        return "system_error"
    
    @staticmethod
    def handle_error(error: Exception, context: str = "", show_technical_details: bool = False) -> Dict[str, Any]:
        """
        Handle error with user-friendly feedback
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            show_technical_details: Whether to show technical error details
            
        Returns:
            Dict with error information and UI components
        """
        error_category = ErrorHandler.classify_error(error)
        error_info = ErrorHandler.ERROR_CATEGORIES[error_category]
        
        # Log the error
        logger.error(f"Error in {context}: {error}", exc_info=True)
        
        # Create error response
        error_response = {
            "error": True,
            "category": error_category,
            "title": error_info["title"],
            "description": error_info["description"],
            "suggestions": error_info["suggestions"],
            "technical_check": error_info["technical_check"],
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "original_error": str(error)
        }
        
        # Display user-friendly error in Streamlit
        ErrorHandler.display_error_ui(error_response, show_technical_details)
        
        return error_response
    
    @staticmethod
    def display_error_ui(error_info: Dict[str, Any], show_technical_details: bool = False):
        """Display user-friendly error UI in Streamlit"""
        
        # Main error message
        st.error(f"{error_info['title']}")
        
        # Error description
        st.markdown(f"**Description:** {error_info['description']}")
        
        # Recovery suggestions
        with st.expander("💡 Suggestions de résolution", expanded=True):
            for i, suggestion in enumerate(error_info["suggestions"], 1):
                st.markdown(f"{i}. {suggestion}")
        
        # System check
        with st.expander("🔍 Vérification système"):
            st.info(f"**Zone à vérifier:** {error_info['technical_check']}")
            
            # Add quick diagnostic buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Réessayer", key="retry_button"):
                    st.experimental_rerun()
            
            with col2:
                if st.button("🧹 Nettoyer Cache", key="clear_cache_button"):
                    st.cache_data.clear()
                    st.success("Cache nettoyé!")
            
            with col3:
                if st.button("📊 État Système", key="system_status_button"):
                    ErrorHandler.show_system_status()
        
        # Technical details (collapsible)
        if show_technical_details:
            with st.expander("🔧 Détails techniques"):
                st.code(f"""
Catégorie: {error_info['category']}
Contexte: {error_info['context']}
Horodatage: {error_info['timestamp']}
Erreur originale: {error_info['original_error']}
                """)
    
    @staticmethod
    def show_system_status():
        """Show system status diagnostics"""
        import psutil
        import os
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🖥️ CPU", f"{cpu_percent:.1f}%")
            
            with col2:
                st.metric("🧠 RAM", f"{memory.percent:.1f}%", 
                         f"{memory.available // (1024**3)} GB libre")
            
            with col3:
                st.metric("💾 Disque", f"{disk.percent:.1f}%",
                         f"{disk.free // (1024**3)} GB libre")
            
            # Service checks
            st.markdown("**État des Services:**")
            
            # Check Ollama
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    st.success("✅ Ollama: Opérationnel")
                else:
                    st.error("❌ Ollama: Problème de connexion")
            except Exception:
                st.error("❌ Ollama: Non accessible")
            
            # Check data directories
            data_dirs = ["data/models", "data/reports", "data/runs"]
            for dir_path in data_dirs:
                if os.path.exists(dir_path):
                    st.success(f"✅ {dir_path}: Accessible")
                else:
                    st.warning(f"⚠️ {dir_path}: Manquant")
                    
        except Exception as e:
            st.error(f"Impossible d'obtenir l'état système: {e}")
    
    @staticmethod
    def create_error_recovery_guide() -> str:
        """Create a comprehensive error recovery guide"""
        guide = """
# 🛠️ Guide de Résolution des Erreurs PlannerIA

## Erreurs Courantes et Solutions

### 1. Problème de Connexion LLM
**Symptômes:** "Connection refused", "Ollama not responding"
**Solutions:**
- Démarrer Ollama: `ollama serve`
- Vérifier le modèle: `ollama list`
- Installer le modèle: `ollama pull llama3.2:latest`

### 2. Erreur de Génération de Plan
**Symptômes:** "Plan generation failed", "Processing error"
**Solutions:**
- Simplifier la description du projet
- Vérifier la syntaxe et la clarté
- Redémarrer l'application

### 3. Problème d'Export
**Symptômes:** "Export failed", "File generation error"
**Solutions:**
- Vérifier l'espace disque
- Fermer les fichiers ouverts
- Réessayer avec un nom différent

### 4. Erreur de Performance
**Symptômes:** Application lente, timeouts
**Solutions:**
- Fermer d'autres applications
- Vérifier RAM et CPU
- Simplifier les demandes

## Contacts Support
- Documentation: Consultez la documentation du projet
- Logs: Vérifiez les logs dans la console
- Redémarrage: Relancez l'application complète
        """
        return guide

# Convenience functions for common error scenarios
def handle_llm_error(error: Exception, context: str = "LLM Operation") -> Dict[str, Any]:
    """Handle LLM-specific errors"""
    return ErrorHandler.handle_error(LLMConnectionError(str(error)), context)

def handle_validation_error(error: Exception, context: str = "Input Validation") -> Dict[str, Any]:
    """Handle validation errors"""
    return ErrorHandler.handle_error(ValidationError(str(error)), context)

def handle_processing_error(error: Exception, context: str = "Plan Processing") -> Dict[str, Any]:
    """Handle processing errors"""
    return ErrorHandler.handle_error(ProcessingError(str(error)), context)

def handle_export_error(error: Exception, context: str = "Export Operation") -> Dict[str, Any]:
    """Handle export errors"""
    return ErrorHandler.handle_error(error, context)

def safe_execute(func, *args, context: str = "", **kwargs) -> Tuple[bool, Any]:
    """
    Safely execute a function with enhanced error handling
    
    Returns:
        Tuple of (success: bool, result: Any)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        error_info = ErrorHandler.handle_error(e, context)
        return False, error_info