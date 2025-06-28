# main.py (FastAPI-Anwendung mit korrigierter Hybrid-Authentifizierung)
# Dateipfad auf Server: /var/www/vhosts/last-strawberry.com/ki_editor_api/app/main.py

# --- Imports ---
import fastapi, uvicorn, shutil, logging, datetime, json, os, secrets
from pathlib import Path
from typing import List, Dict, Annotated, Optional, Generator, Any
from fastapi import Depends, HTTPException, status, Header
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import timedelta

# --- Konfiguration laden ---
try:
    import server_config
    SECRET_KEY = getattr(server_config, 'SECRET_KEY', 'BITTE_SEHR_SICHEREN_SECRET_KEY_IN_CONFIG_SETZEN')
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = getattr(server_config, 'ACCESS_TOKEN_EXPIRE_MINUTES', 60)
    DATABASE_URL = getattr(server_config, 'DATABASE_URL', "sqlite:///./ki_editor_main.db")
    BATCH_UPLOAD_DIR_STR = getattr(server_config, 'BATCH_UPLOAD_PATH', './data_batches/')
    MODEL_UPLOAD_DIR_STR = getattr(server_config, 'MODEL_UPLOAD_PATH', './uploaded_models/')
    CONFIG_LOADED = True
except (ImportError, AttributeError) as e:
    logging.critical(f"FEHLER: server_config.py nicht gefunden oder fehlerhaft: {e}")
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES = "fallback_secret", "HS256", 5
    DATABASE_URL, BATCH_UPLOAD_DIR_STR, MODEL_UPLOAD_DIR_STR = "sqlite:///./fallback.db", "./fb_data/", "./fb_models/"
    CONFIG_LOADED = False

# --- Pfad-Konfiguration, Logging ---
BATCH_UPLOAD_DIR = Path(BATCH_UPLOAD_DIR_STR); PROCESSED_BATCH_DIR = BATCH_UPLOAD_DIR / "processed_by_tower"; MODEL_UPLOAD_DIR = Path(MODEL_UPLOAD_DIR_STR)
BATCH_UPLOAD_DIR.mkdir(parents=True, exist_ok=True); PROCESSED_BATCH_DIR.mkdir(parents=True, exist_ok=True); MODEL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Sicherheit & Datenbank ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"; id = Column(Integer, primary_key=True, index=True); username = Column(String, unique=True, index=True, nullable=False); password_hash = Column(String, nullable=True); api_key_hash = Column(String, nullable=True); roles = Column(String, nullable=False, default="ROLE_DATA_CONTRIBUTOR,ROLE_MODEL_CONSUMER"); is_active = Column(Boolean, default=True); created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)); notes = Column(String, nullable=True)
Base.metadata.create_all(bind=engine)

# --- Pydantic-Schemas ---
class UserInDB(BaseModel): id: int; username: str; roles: str; is_active: bool; created_at: datetime.datetime; notes: Optional[str] = None
class Config: from_attributes = True
class Token(BaseModel): access_token: str; token_type: str
class StatusUpdate(BaseModel): is_active: bool
class RolesUpdate(BaseModel): roles: str
class UserCreateWithPassword(BaseModel): username: str; password: str; roles: str; notes: Optional[str] = None

# --- Authentifizierung & Hilfsfunktionen ---
def get_db():
    db = SessionLocal()
    _ = db.query(User).first()
    try:
        yield db
    finally:
        db.close()

def verify_password(p, h): return pwd_context.verify(p, h)
def get_password_hash(p): return pwd_context.hash(p)
def verify_api_key(p, h): return pwd_context.verify(p, h)
def get_api_key_hash(p): return pwd_context.hash(p)
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/token")

async def get_current_active_user_by_token(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)) -> User:
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]); username: str = payload.get("sub")
        if username is None: raise exc
        user_id = payload.get("uid")
        if user_id is None: raise exc
    except JWTError: raise exc
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if user is None: raise exc
    return user

async def get_current_active_user_by_api_key(api_key: Annotated[str, Header(alias="X-API-Key")], db: Session = Depends(get_db)) -> User:
    if not api_key: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API-Key fehlt im Header (X-API-Key)")
    users = db.query(User).filter(User.is_active == True, User.api_key_hash.isnot(None)).all()
    for u in users:
        if u.api_key_hash and verify_api_key(api_key, u.api_key_hash): return u
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ungültiger API-Key")

def require_role(required_roles: List[str], auth_method: str = "token"):
    async def role_checker(user: User = Depends(get_current_active_user_by_token if auth_method == "token" else get_current_active_user_by_api_key)):
        user_roles = {role.strip() for role in user.roles.split(',')}
        if "ROLE_USER_ADMIN" in user_roles: return user # Admin darf alles
        if not any(req_role in user_roles for req_role in required_roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Benötigt eine der Rollen: {', '.join(required_roles)}")
        return user
    return role_checker

# --- FastAPI App & Router ---
app = fastapi.FastAPI(title="KI-Editor API mit Benutzerverwaltung")
admin_router = fastapi.APIRouter(prefix="/admin", tags=["Admin - Benutzerverwaltung"])

# --- Admin-Endpunkte ---
@admin_router.post("/token", response_model=Token, summary="Login für Web-UI Admins")
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username, User.is_active == True).first()
    if not user or not user.password_hash or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Falscher Benutzername oder Passwort")
    user_roles = {role.strip() for role in user.roles.split(',')}
    if "ROLE_USER_ADMIN" not in user_roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nur Admins können sich im Web-Dashboard einloggen")
    token = create_access_token(data={"sub": user.username, "roles": list(user_roles), "uid": user.id})
    return {"access_token": token, "token_type": "bearer"}

@admin_router.get("/users", response_model=List[UserInDB], dependencies=[Depends(require_role(["ROLE_USER_ADMIN"]))])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(User).offset(skip).limit(limit).all()

@admin_router.post("/users", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_role(["ROLE_USER_ADMIN"]))])
def create_user(user_data: UserCreateWithPassword, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user_data.username).first(): raise HTTPException(status_code=400, detail="Benutzername bereits registriert")
    password_hash = get_password_hash(user_data.password); new_api_key = secrets.token_urlsafe(32); api_key_hash = get_api_key_hash(new_api_key)
    db_user = User(username=user_data.username, password_hash=password_hash, api_key_hash=api_key_hash, roles=user_data.roles, notes=user_data.notes)
    db.add(db_user); db.commit(); db.refresh(db_user)
    logger.info(f"Admin hat neuen Benutzer erstellt: '{user_data.username}'"); return {"message": "Benutzer erfolgreich erstellt.", "user_id": db_user.id, "username": db_user.username, "api_key": new_api_key}

@admin_router.put("/users/{user_id}/status", response_model=UserInDB, dependencies=[Depends(require_role(["ROLE_USER_ADMIN"]))])
def update_user_status(user_id: int, status_update: StatusUpdate, current_user: Annotated[User, Depends(get_current_active_user_by_token)], db: Session = Depends(get_db)):
    if user_id == 1 and current_user.id != 1: raise HTTPException(status_code=403, detail="Haupt-Admin (ID 1) kann nicht von anderen geändert werden.")
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    db_user.is_active = status_update.is_active; db.commit(); db.refresh(db_user)
    logger.info(f"Admin '{current_user.username}' hat Status von '{db_user.username}' auf '{db_user.is_active}' gesetzt."); return db_user

@admin_router.put("/users/{user_id}/roles", response_model=UserInDB, dependencies=[Depends(require_role(["ROLE_USER_ADMIN"]))])
def update_user_roles(user_id: int, roles_update: RolesUpdate, current_user: Annotated[User, Depends(get_current_active_user_by_token)], db: Session = Depends(get_db)):
    if user_id == 1 and current_user.id != 1: raise HTTPException(status_code=403, detail="Rollen des Haupt-Admins können nicht von anderen geändert werden.")
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    db_user.roles = roles_update.roles; db.commit(); db.refresh(db_user)
    logger.info(f"Admin '{current_user.username}' hat Rollen von '{db_user.username}' auf '{db_user.roles}' gesetzt."); return db_user

@admin_router.put("/users/{user_id}/reset-password", response_model=Dict[str, str], dependencies=[Depends(require_role(["ROLE_USER_ADMIN"]))])
def reset_user_password(user_id: int, current_user: Annotated[User, Depends(get_current_active_user_by_token)], db: Session = Depends(get_db)):
    if user_id == 1 and current_user.id != 1: raise HTTPException(status_code=403, detail="Passwort des Haupt-Admins kann nicht zurückgesetzt werden.")
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    new_password = secrets.token_urlsafe(16); db_user.password_hash = get_password_hash(new_password); db.commit()
    logger.info(f"Admin '{current_user.username}' hat Passwort für '{db_user.username}' zurückgesetzt."); return {"message": "Passwort erfolgreich zurückgesetzt.", "username": db_user.username, "new_password": new_password}

@admin_router.put("/users/{user_id}/reset-apikey", response_model=Dict[str, str], dependencies=[Depends(require_role(["ROLE_USER_ADMIN"]))])
def reset_user_apikey(user_id: int, current_user: Annotated[User, Depends(get_current_active_user_by_token)], db: Session = Depends(get_db)):
    if user_id == 1 and current_user.id != 1: raise HTTPException(status_code=403, detail="API-Key des Haupt-Admins kann nicht zurückgesetzt werden.")
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    new_api_key = secrets.token_urlsafe(32); db_user.api_key_hash = get_api_key_hash(new_api_key); db.commit()
    logger.info(f"Admin '{current_user.username}' hat API-Key für '{db_user.username}' zurückgesetzt."); return {"message": "API-Key erfolgreich zurückgesetzt.", "username": db_user.username, "new_api_key": new_api_key}


app.include_router(admin_router)

# --- Haupt-API-Endpunkte (verwenden API-Key-Auth) ---
def require_api_role(required_roles: List[str]):
    async def role_checker(user: User = Depends(get_current_active_user_by_api_key)):
        user_roles = {role.strip() for role in user.roles.split(',')}
        if "ROLE_USER_ADMIN" in user_roles: return user
        if not any(req_role in user_roles for req_role in required_roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"API-Key hat keine Berechtigung. Benötigt: {', '.join(required_roles)}")
        return user
    return role_checker

@app.post("/data/upload_batch", dependencies=[Depends(require_role(["ROLE_DATA_CONTRIBUTOR"], auth_method="apikey"))])
async def upload_data_batch_api(client_id: Annotated[str, fastapi.Query()], batch_file: Annotated[fastapi.UploadFile, fastapi.File()], current_user: Annotated[User, Depends(get_current_active_user_by_api_key)]):
    # Logik wie zuvor
    logger.info(f"Benutzer (via API-Key) '{current_user.username}' lädt Daten-Batch hoch.")
    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_client_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in client_id)
        filename = f"batch_user_{current_user.id}_{safe_client_id}_{timestamp}.jsonl"
        file_path = BATCH_UPLOAD_DIR / filename
        with file_path.open("wb") as buffer: shutil.copyfileobj(batch_file.file, buffer)
        return {"message": "Daten-Batch erfolgreich hochgeladen.", "filename_on_server": filename}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: await batch_file.close()

# (Weitere Endpunkte wie /model/upload etc. müssen ebenfalls umgestellt werden)

# --- Admin-Web-UI Endpunkt ---
@app.get("/admin-dashboard", response_class=HTMLResponse)
async def get_admin_dashboard():
    admin_page_path = Path(__file__).parent / "admin.html"
    if not admin_page_path.exists():
        raise HTTPException(status_code=404, detail="Admin-Dashboard HTML-Datei nicht gefunden.")
    return FileResponse(admin_page_path)

@app.get("/admin.js", include_in_schema=False)
async def get_admin_js():
    """Liefert die JavaScript-Datei für das Admin-Dashboard."""
    js_path = Path(__file__).parent / "admin.js"
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="admin.js nicht gefunden.")
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Liefert das Favicon für die Weboberfläche."""
    favicon_path = Path(__file__).parent / "favicon.ico"
    if favicon_path.is_file():
        return FileResponse(favicon_path)
    else:
        # Gibt 404 zurück, wenn keine Favicon-Datei gefunden wird
        return fastapi.Response(status_code=status.HTTP_404_NOT_FOUND)

if __name__ == "__main__": uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
