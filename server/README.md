
# Signz-Mind – Server

**Backend-API & Admin-Web-UI für Benutzerverwaltung, Datenmanagement und Modellbereitstellung**

---

## Überblick

Das **Server-Modul** ist das zentrale Backend von Signz-Mind.  
Es stellt folgende Kernfunktionen bereit:

- Benutzer- und Rollenverwaltung (inkl. API-Key & Passwort-Reset)
- Upload/Download von Trainingsdaten-Batches (API)
- Modellverwaltung und Modell-Uploads (API)
- Admin-Web-UI (Benutzer- und Rechteverwaltung im Browser)
- Sichere Authentifizierung (JWT, API-Key)
- Robustes Logging & Fehlerhandling

---

## Features

- Moderne, sichere FastAPI-Architektur
- SQLite-Datenbank (kann einfach auf PostgreSQL/MySQL erweitert werden)
- Multi-User, feingranulare Rollen, Admin-Schutz für Hauptadmin
- Admin-UI mit TailwindCSS, JWT-basiertem Login, Passwort-/API-Key-Reset
- Automatische Anlage notwendiger Verzeichnisse
- OpenAPI/Swagger-Dokumentation für alle REST-Endpoints

---

## Installation & Setup

1. **Voraussetzungen**
   - Python 3.10+
   - (Optional) Virtuelle Umgebung:  
     ```
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```
   - (Optional) Gunicorn/Uvicorn für Produktion

2. **Initiales Setup**
   - Führe das Script `/server/setup_server.py` aus:
     ```
     python setup_server.py
     ```
     → Erstellt initiale Konfigurationsdatei und Admin-User

3. **Starten der API**
   - Lokal (Entwicklung):
     ```
     uvicorn main:app --reload
     ```
   - Produktion (Beispiel, Gunicorn+UvicornWorker):
     ```
     gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
     ```
   - Admin-UI erreichbar unter:  
     `http(s)://<server>/admin-dashboard`

---

## Verzeichnis & Module

- `main.py` – FastAPI-Anwendung (API, Auth, Endpunkte)
- `setup_server.py` – Initiales Setup (Config, Admin-Benutzer, DB)
- `server_config.py` – generiert, enthält sensible Einstellungen (SECRET_KEY etc.)
- `admin.html`, `admin.js` – Web-UI (Benutzerverwaltung)
- Uploadverzeichnisse (`data_batches/`, `models/`) werden automatisch angelegt

---

## Security & Hinweise

- **SECRET_KEY und Datenbankpfad** werden in `server_config.py` generiert – NIEMALS öffentlich machen!
- **API-Keys & Passwörter** werden nur gehasht gespeichert.
- **Admin-User (ID 1)** ist besonders geschützt und kann nicht von anderen geändert werden.
- **Backups:** Regelmäßige Sicherungen der DB und Upload-Ordner empfohlen.

---

## Weiteres & Support

- Featurewünsche und Bugreports bitte im [Hauptprojekt-Repo](../README.md).
- Für produktive Nutzung: Reverse-Proxy (z.B. nginx) und HTTPS empfohlen.

---

## Project Summary in English

The **Signz-Mind server** is the central backend, providing FastAPI endpoints for user management, data and model upload/download, and a modern admin web UI.  
It features role-based security (JWT and API-keys), supports multiple users, and is designed for professional environments with robust logging and modularity.  
All configuration is handled via a secure setup script, and the web UI allows full management of users and roles.

---
