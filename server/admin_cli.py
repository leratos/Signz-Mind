# admin_cli.py
# Ein Kommandozeilen-Tool zur Verwaltung von Benutzern auf dem Server.
# Setzt eine bereits existierende und konfigurierte API voraus.

import requests
import argparse
import json
import getpass

def create_user(server_url: str, admin_key: str, username: str, roles: str):
    """Erstellt einen neuen Benutzer."""
    api_url = f"{server_url.rstrip('/')}/admin/users/"
    headers = {"X-Admin-API-Key": admin_key, "Content-Type": "application/json"}
    payload = {"username": username, "roles": roles, "is_active": True}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        print("\n--- Benutzer erfolgreich erstellt! ---")
        print(f"  Benutzer-ID: {data.get('user_id')}")
        print(f"  Benutzername: {data.get('username')}")
        print("\n!!! WICHTIG: API-Key wird nur EINMALIG angezeigt. Sicher kopieren und an den Benutzer weitergeben. !!!")
        print(f"  NEUER API-KEY: {data.get('api_key')}\n")
    except requests.exceptions.HTTPError as e:
        print(f"\nFehler: {e.response.status_code}")
        try:
            print(f"Server-Nachricht: {e.response.json().get('detail')}")
        except json.JSONDecodeError:
            print(f"Server-Antwort (nicht JSON): {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\nVerbindungsfehler: {e}")

def list_users(server_url: str, admin_key: str):
    """Listet vorhandene Benutzer auf."""
    api_url = f"{server_url.rstrip('/')}/admin/users/"
    headers = {"X-Admin-API-Key": admin_key}
    
    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        users = response.json()
        print("\n--- Vorhandene Benutzer ---")
        if not users:
            print("Keine Benutzer gefunden.")
            return
        for user in users:
            print(f"  ID: {user['id']:<4} | Benutzername: {user['username']:<20} | Aktiv: {str(user['is_active']):<5} | Rollen: {user['roles']}")
    except requests.exceptions.HTTPError as e:
        print(f"\nFehler: {e.response.status_code}")
        try:
            print(f"Server-Nachricht: {e.response.json().get('detail')}")
        except json.JSONDecodeError:
            print(f"Server-Antwort (nicht JSON): {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\nVerbindungsfehler: {e}")

def main():
    parser = argparse.ArgumentParser(description="Admin-CLI für KI-Editor Benutzerverwaltung.")
    parser.add_argument("--url", required=True, help="Basis-URL der API (z.B. https://api.last-strawberry.com)")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Verfügbare Befehle")

    parser_create = subparsers.add_parser("create", help="Erstellt einen neuen Benutzer.")
    parser_create.add_argument("username", help="Der eindeutige Benutzername.")
    parser_create.add_argument("roles", help="Komma-separierte Liste von Rollen (z.B. 'ROLE_DATA_CONTRIBUTOR,ROLE_MODEL_CONSUMER')")

    parser_list = subparsers.add_parser("list", help="Listet alle Benutzer auf.")
    
    args = parser.parse_args()
    
    admin_api_key = getpass.getpass("Bitte Admin-API-Key (aus server_config.py) eingeben: ")

    if args.command == "create":
        create_user(args.url, admin_api_key, args.username, args.roles)
    elif args.command == "list":
        list_users(args.url, admin_api_key)

if __name__ == "__main__":
    main()
