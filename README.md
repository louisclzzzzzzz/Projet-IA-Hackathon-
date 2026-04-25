# Hackathon: Prédiction de Séries Temporelles Financières

Bienvenue sur le dépôt du projet IA de prédiction de séries temporelles financières. Ce projet a été développé dans le cadre d'un Hackathon ciblant la prévision intraday et multijournalière.

## 🎯 Objectif du Projet
L'objectif est d'utiliser l'Architecture Deep Learning pour construire un modèle robuste prédisant l'évolution sur **15 pas de temps futurs** de deux indicateurs essentiels en trading quantitatif :
1. **MACD(12, 26, 9)** : Moving Average Convergence Divergence
2. **Stochastique %K(14, 3, 5)** : Oscillateur de momentum

## ⚙️ Approches Évaluées
Le pipeline implémenté évalue systématiquement et de manière modulaire 6 méthodes de prédiction croissante :
- **Baseline (Random Walk)** : Repère linéaire simple basé sur l'historique récent.
- **ARIMA (5, 1, 2)** : Algorithme classique en *rolling forecast*.
- **LSTM** : Réseau de neurones récurrents avec état de cellule long-terme pour capturer les tendances longues de forte amplitude.
- **GRU** : Variantes compactes des RNNs surmontant rapidement le problème d'évanescence du gradient, idéal pour un momentum rapide.
- **Transformer Encoder** : Mécanisme d'auto-attention s'affranchissant des récurrences purement temporelles.
- **Ensemble (Nelder-Mead)** : Modèle d'optimisation robuste qui moyenne pondère les résultats LSTM, GRU et Transformer.

## 📊 Actifs et Datasets Testés
Ce framework a été construit pour être universel, ne se limitant pas à l'optimisation locale mais gérant toute dynamique boursière. Les benchmarks ont été opérés sur :
- **Apple (AAPL)** : Modèle Intraday - barres 5 minutes. (Forte profondeur de données, volatilité de fond régulière).
- **Microsoft (MSFT)** : Modèle Journalier - $\approx$ 1 200 jours. (Faible profondeur de données, grand retournement sur la période observée).
- **Nvidia (NVDA)** : Modèle Journalier - $\approx$ 1 250 jours. (Haute dérive directionnelle et forte variation relative).

## 🚀 Comment lancer le pipeline

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
2. **Lancer le pipeline d'entraînement** (exemple pour Nvidia) :
   ```bash
   python pipeline.py --folder data --ticker NVDA
   ```
   Ce script chargera le fichier `NVDA_data.xlsx` depuis le dossier `data/`, préparera les séquences, entrainera les 5 grands modèles, et extraira ses graphes dans le dossier `/figures` ainsi que les KPIs d'évaluation dans `/results`.

## 📂 Structure du projet
- **/data** : Jeux de données financiers (Excel).
- **/docs** : Code source LaTeX et PDF du rapport final.
- **/figures** : Visualisations et graphes d'évaluation.
- **/models** : Architectures deep learning et statistiques python.
- **/preprocessing** : Modules de nettoyage et calcul de features.
- **/results** : Fichiers JSON et CSV contenant les scores des expérimentations et rapports Markdown.
- **/scripts** : Utilitaires de test ou de minage isolés.

## 📈 Résultats et Analyse (Rapport)
Le détail complet des performances et de la métrique $S \ge 0.55$ du hackathon est disponible dans le fichier compilé `rapport.pdf`.
Le **GRU** s'impose sur les prédictions Intraday statiques (Apple), tandis que le **LSTM** est indispensable pour dompter les hausses paraboliques récentes sur le MACD en journalier (Nvidia). Le **Transformer**, bien qu'expérimental et limitant car ne tournant que sur un format CPU ici, performe de façon surprenante sur le signal pur du stochastique (Microsoft).

---
> 👩‍💻 *Auteur : Louis Cluzel -- M1 Intelligence Artificielle (Université Clermont Auvergne)*
