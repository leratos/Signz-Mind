
# Signz-Mind – Tower (Cronjob)

**Automatisierter Trainingsrechner für tägliches Fine-Tuning und Modell-Upload**

---

## Überblick

Das **Tower-Modul** ist der Trainings- und Workflow-Knoten von Signz-Mind.  
Es läuft als täglicher Cronjob (bzw. geplante Aufgabe) und führt folgende Aufgaben automatisiert aus:

- Prüft, ob neue Trainingsdaten auf dem Server verfügbar sind
- Importiert und verarbeitet Datenbatches
- Startet Fine-Tuning eines lokalen KI-Modells (z.B. QLoRA)
- Verpackt und lädt neue Modelladapter zurück zum Server

> **Hinweis:** Dieses Modul läuft typischerweise auf einer separaten, leistungsfähigen Workstation ("Tower"), kann aber auch auf jedem anderen Rechner mit GPU ausgeführt werden.

---

## Features

- Automatisiertes Training per Zeitplan (z.B. täglich um 13 Uhr)
- Robustes Logging aller Schritte und Fehlerfälle
- Konfigurierbar über JSON-Config
- Upload von Modellen inklusive Metadaten an die zentrale Server-API
- Abgleich und Markierung verarbeiteter Datenbatches

---

## Installation & Einrichtung

1. **Voraussetzungen**
   - Python 3.10+ (Empfohlen: venv)
   - Abhängigkeiten:  
     ```
     pip install -r requirements.txt
     ```
   - GPU-Treiber/Umgebung, falls Fine-Tuning genutzt wird (PyTorch, transformers, peft, usw.)

2. **Konfiguration**
   - Kopiere `/cronjob/app_settings_tower.json.example` nach `app_settings_tower.json` und passe folgende Werte an:
     - `server_api_url`: URL zur Server-API (z.B. https://api.example.com)
     - `tower_api_key`: API-Key für diesen Tower (vom Admin generiert)
     - Weitere Hyperparameter optional anpassbar

3. **Cronjob/Scheduler einrichten**
   - Beispiel (Linux-Cronjob, täglich 13 Uhr):
     ```
     0 13 * * * /usr/bin/python3 /pfad/zum/repo/cronjob/tower_daily_workflow.py
     ```
   - Alternativ: Aufgabenplanung unter Windows nutzen

---

## Verzeichnis & Module

- `tower_daily_workflow.py` – Haupt-Workflow (ruft alle Teilprozesse auf)
- `tower_data_processor.py` – Download & Import neuer Datenbatches
- `tower_model_uploader.py` – Verpackung und Upload trainierter Modelle
- `app_settings_tower.json` – Konfigurationsdatei
- Logs und temporäre Verzeichnisse werden automatisch angelegt

---

## Troubleshooting

- **Fehler beim Import lokaler Module:**  
  Stelle sicher, dass der Arbeitsordner korrekt gesetzt ist und das Repo als Paket ausführbar ist (`PYTHONPATH` ggf. anpassen).
- **Fehler beim Modelltraining:**  
  Prüfe GPU-Auslastung, Speicherplatz und die installierten ML-Abhängigkeiten.
- **Verbindungsprobleme zum Server:**  
  Prüfe die API-URL, API-Key und Netzwerkzugang.

---

## Kontakt & Support

Featurewünsche, Bugreports und Support bitte über das Hauptprojekt auf GitHub.

---

## Project Summary in English

**Tower** is the automated training and workflow node of Signz-Mind.  
It checks the server for new data batches, imports and processes them, triggers daily fine-tuning of local AI models, and uploads updated adapters to the server.  
Designed for scheduled execution (e.g., via cron), it provides robust logging, configuration, and is optimized for use on a dedicated GPU workstation.

---
