#!/usr/bin/env python3
"""
Petit test pour vérifier que CrewAI utilise bien ton modèle Ollama local.
"""

from crewai.llm import LLM

def main():
    # ✅ on appelle directement ton modèle local
    llm = LLM(model="ollama/heavylildude/magnus:latest")

    # Prompt simple
    prompt = "Écris une fonction Python qui calcule la factorielle d’un nombre."

    print("=== TEST LLM LOCAL ===")
    print(f"Prompt envoyé : {prompt}\n")
    response = llm.call(prompt)
    print("Réponse du modèle :\n")
    print(response)

if __name__ == "__main__":
    main()
