# gfsapi.py
"""
Full backend with:
- Firebase Firestore user system (email, password, API keys)
- Multi-store per user (display_name, actual_store_name)
- Per-API-key 1GB quota
- read (boolean) to control RAG accessibility
- Make-Live returning only final Ask-URL string
- Ask endpoint using /{user_id}/{store}?ask=...
- Temporary upload deletion (documents never stored)
"""

import os
import time
import json
import shutil
import re
import uuid
import requests
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from fastapi import Body
# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Gemini
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ---------------- CONFIG ----------------
DATA_FILE = "/data/gemini_stores.json"
UPLOAD_ROOT = Path("/data/uploads")

MAX_FILE_BYTES = 50 * 1024 * 1024                   # 50MB per file
MAX_TOTAL_BYTES_PER_API_KEY = 1024 * 1024 * 1024    # 1GB per API key

POLL_INTERVAL = 2
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# BASE_URL provided through environment variable
BASE_URL = os.getenv("BASE_URL")

# ----------------------------------------

# Firebase initialization
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI(title="Gemini File Search RAG Backend Full System")

# ---------------- Helpers for local metadata ----------------


def verify_gemini_key(api_key: str) -> bool:
    try:
        client = genai.Client(api_key=api_key)

        # Try generating something small using the SAME model (2.5-flash)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )

        # If no exception → key is valid
        return True

    except Exception as e:
        # Invalid model? invalid key? quota? → treat as invalid
        return False


def ensure_dirs():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    Path(DATA_FILE).parent.mkdir(parents=True, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"file_stores": {}, "current_store_name": None}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

ensure_dirs()
if not os.path.exists(DATA_FILE):
    save_data({"file_stores": {}, "current_store_name": None})


# ---------------- Request Models ----------------

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AddApiKeyRequest(BaseModel):
    api_key: str

class DeleteApiKeyRequest(BaseModel):
    api_key: str

class CreateStoreRequest(BaseModel):
    user_id: str
    display_name: str


# ---------------- Gemini Helpers ----------------

def init_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError("google-genai SDK missing")
    return genai.Client(api_key=api_key)

def wait_for_operation(client, operation):
    op = operation
    while not getattr(op, "done", False):
        time.sleep(POLL_INTERVAL)
        try:
            op = client.operations.get(op)
        except Exception:
            pass

    if getattr(op, "error", None):
        raise RuntimeError(str(op.error))

    return op

def rest_list_documents_for_store(store_name: str, api_key: str):
    url = f"{GEMINI_REST_BASE}/{store_name}/documents"
    try:
        resp = requests.get(url, params={"key": api_key}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("documents", [])
    except Exception:
        return []

# ---------------- Filename Sanitization ----------------

def clean_filename(name: str, max_len=150) -> str:
    if not name:
        return "file"
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"^\.+", "", name)
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name)
    name = re.sub(r"__+", "_", name)
    if len(name) > max_len:
        name = name[:max_len]
    return name or "file"

# ---------------- Quota Helper ----------------

def compute_api_key_usage_bytes(user_id: str, api_key: str):
    data = load_data()
    total = 0
    for store in data["file_stores"].values():
        if store.get("user_id") == user_id and store.get("api_key") == api_key:
            for f in store.get("files", []):
                total += int(f.get("size_bytes", 0))
    return total

def pick_api_key_for_new_store(user_id: str) -> str:
    doc = db.collection("users").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    api_keys = doc.to_dict().get("apiKeys", [])
    if not api_keys:
        raise HTTPException(status_code=400, detail="User has no API keys")

    for entry in api_keys:
        key = entry.get("key")
        if compute_api_key_usage_bytes(user_id, key) < MAX_TOTAL_BYTES_PER_API_KEY:
            return key

    raise HTTPException(status_code=400, detail="All API keys reached 1GB limit.")


# ============================================================
# USERS
# ============================================================

@app.post("/users/register")
def register_user(payload: RegisterRequest):
    existing = list(db.collection("users").where("email", "==", payload.email).limit(1).stream())
    if existing:
        return JSONResponse({"error": "Email already registered"}, 400)

    user_id = uuid.uuid4().hex
    db.collection("users").document(user_id).set({
        "email": payload.email,
        "password": payload.password,
        "apiKeys": [],
        "createdAt": time.strftime("%Y-%m-%d %H:%M:%S")

    })

    return {"success": True, "user_id": user_id}


@app.post("/users/login")
def user_login(payload: LoginRequest):
    docs = list(db.collection("users")
                .where("email", "==", payload.email)
                .limit(1).stream())
    if not docs:
        return JSONResponse({"error": "Invalid email or password"}, 401)

    doc = docs[0].to_dict()
    if doc.get("password") != payload.password:
        return JSONResponse({"error": "Invalid email or password"}, 401)

    return {"success": True, "user_id": docs[0].id}


# ============================================================
# API KEY MANAGEMENT
# ============================================================

@app.post("/users/{user_id}/api-keys/add")
def add_api_key(user_id: str, payload: AddApiKeyRequest = Body(...)):
    ref = db.collection("users").document(user_id)
    doc = ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate Gemini API key
    is_valid = verify_gemini_key(payload.api_key)
    if not is_valid:
        return JSONResponse({"error": "Invalid Gemini API key"}, status_code=400)

    user = doc.to_dict() or {}
    keys = user.get("apiKeys")

    if not isinstance(keys, list):
        keys = []

    # Duplicate check
    for item in keys:
        if item.get("key") == payload.api_key:
            return JSONResponse({"error": "API key exists"}, 400)

    # Add key
    keys.append({
        "key": payload.api_key,
        "createdAt": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    ref.set({"apiKeys": keys}, merge=True)

    return {"success": True, "apiKeyCount": len(keys)}



@app.delete("/users/{user_id}/api-keys/delete")
def delete_api_key(user_id: str, payload: DeleteApiKeyRequest = Body(...)):
    ref = db.collection("users").document(user_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(404, "User not found")

    user = doc.to_dict() or {}
    keys = user.get("apiKeys")

    # Ensure apiKeys is a list
    if not isinstance(keys, list):
        keys = []

    # Filter out the key
    new_keys = [k for k in keys if k.get("key") != payload.api_key]

    if len(new_keys) == len(keys):
        return JSONResponse({"error": "API key not found"}, 404)

    ref.update({"apiKeys": new_keys})

    return {"success": True, "apiKeyCount": len(new_keys)}



# ============================================================
# STORE CREATE
# ============================================================

@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    user_id = payload.user_id

    api_key = pick_api_key_for_new_store(user_id)
    client = init_gemini_client(api_key)

    suffix = uuid.uuid4().hex[:8]
    actual_store = f"{clean_filename(payload.display_name)}_{suffix}"

    try:
        fs_store = client.file_search_stores.create(
            config={"display_name": payload.display_name}
        )
        fs_name = getattr(fs_store, "name", None) or fs_store
    except Exception as e:
        raise HTTPException(500, f"Gemini store create failed: {e}")

    data = load_data()
    data["file_stores"][actual_store] = {
        "store_name": actual_store,
        "display_name": payload.display_name,
        "file_search_store_name": fs_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [],
        "user_id": user_id,
        "api_key": api_key,
        "read": False
    }

    save_data(data)

    return {
        "success": True,
        "store_actual": actual_store,
        "display_name": payload.display_name
    }


# ============================================================
# TOGGLE READ FLAG
# ============================================================

@app.post("/stores/{user_id}/{store_name}/toggle-read")
def toggle_read(user_id: str, store_name: str):
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]
    if meta["user_id"] != user_id:
        raise HTTPException(403, "Not your store")

    meta["read"] = not bool(meta.get("read", False))
    data["file_stores"][store_name] = meta
    save_data(data)

    return {"success": True, "read": meta["read"]}


# ============================================================
# MAKE-LIVE → RETURNS ONLY FINAL URL STRING
# ============================================================

@app.get("/stores/{user_id}/{store_name}/make-live", response_class=PlainTextResponse)
def make_live(user_id: str, store_name: str):
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]

    if meta["user_id"] != user_id:
        raise HTTPException(403, "Store does not belong to this user")

    if not meta.get("read", False):
        raise HTTPException(400, "Store is not live. Toggle read first.")

    if not BASE_URL:
        raise HTTPException(500, "BASE_URL not set in environment")

    # REQUIRED FORMAT:
    # https://BASE_URL/{user_id}/{store_name}?ask={question}
    final_url = f"{BASE_URL}/{user_id}/{store_name}?ask={{question}}"

    # Return EXACT plain text URL
    return final_url


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

@app.post("/stores/{user_id}/{store_name}/upload")
async def upload_docs(
    user_id: str,
    store_name: str,
    limit: bool = Form(True),
    files: List[UploadFile] = File(...)
):
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]

    if meta["user_id"] != user_id:
        raise HTTPException(403, "Not your store")

    fs_name = meta["file_search_store_name"]
    api_key = meta["api_key"]

    client = init_gemini_client(api_key)

    temp_folder = UPLOAD_ROOT / store_name
    temp_folder.mkdir(parents=True, exist_ok=True)

    results = []

    current_usage = compute_api_key_usage_bytes(user_id, api_key)

    for f in files:
        original = f.filename or "file"
        filename = clean_filename(original)
        temp_path = temp_folder / filename

        size = 0
        try:
            async with aiofiles.open(temp_path, "wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > MAX_FILE_BYTES:
                        await out.close()
                        os.remove(temp_path)
                        results.append({
                            "filename": filename,
                            "uploaded": False,
                            "reason": "File exceeds 50MB limit"
                        })
                        break
                    await out.write(chunk)
        except Exception as e:
            results.append({"filename": filename, "uploaded": False, "reason": str(e)})
            continue

        if size + current_usage > MAX_TOTAL_BYTES_PER_API_KEY:
            os.remove(temp_path)
            results.append({
                "filename": filename,
                "uploaded": False,
                "reason": "1GB API key quota exceeded"
            })
            continue

        # ---- Gemini Upload ----
        document_resource = None
        document_id = None

        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file=str(temp_path),
                file_search_store_name=fs_name,
                config={"display_name": filename}
            )
            op = wait_for_operation(client, op)

            try:
                document_resource = op.response.file_search_document.name
            except:
                document_resource = None

            if not document_resource:
                docs = rest_list_documents_for_store(fs_name, api_key)
                for d in docs:
                    if d.get("displayName") == filename:
                        document_resource = d.get("name")
                        break

            if document_resource:
                document_id = document_resource.split("/")[-1]
        except Exception as e:
            results.append({
                "filename": filename,
                "uploaded": True,
                "indexed": False,
                "error": str(e)
            })
            os.remove(temp_path)
            continue

        # delete temp
        os.remove(temp_path)

        meta["files"].append({
            "display_name": filename,
            "size_bytes": size,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_resource": document_resource,
            "document_id": document_id
        })

        current_usage += size
        results.append({
            "filename": filename,
            "uploaded": True,
            "indexed": True,
            "document_id": document_id
        })

    data["file_stores"][store_name] = meta
    save_data(data)

    return {"success": True, "results": results}


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/stores/{user_id}/{store_name}/documents/{doc_id}")
def delete_doc(user_id: str, store_name: str, doc_id: str):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]
    if meta["user_id"] != user_id:
        raise HTTPException(403, "Not your store")

    fs_name = meta["file_search_store_name"]
    api_key = meta["api_key"]

    url = f"{GEMINI_REST_BASE}/{fs_name}/documents/{doc_id}"
    resp = requests.delete(url, params={"force": "true", "key": api_key})

    if resp.status_code not in (200, 204):
        return JSONResponse({"error": resp.text}, resp.status_code)

    meta["files"] = [x for x in meta["files"] if x.get("document_id") != doc_id]
    data["file_stores"][store_name] = meta
    save_data(data)

    return {"success": True}


# ============================================================
# DELETE STORE
# ============================================================

@app.delete("/stores/{user_id}/{store_name}")
def delete_store(user_id: str, store_name: str):
    data = load_data()
    if store_name not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store_name]

    if meta["user_id"] != user_id:
        raise HTTPException(403, "Not your store")

    fs_name = meta["file_search_store_name"]
    api_key = meta["api_key"]

    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=fs_name, config={"force": True})
    except:
        pass

    # delete temp
    folder = UPLOAD_ROOT / store_name
    if folder.exists():
        shutil.rmtree(folder)

    del data["file_stores"][store_name]
    save_data(data)

    return {"success": True}

# ============================================================
# LIST ALL DOCUMENTS IN A STORE
# ============================================================

@app.get("/stores/{user_id}/{store_name}/documents")
def list_documents(user_id: str, store_name: str):
    """
    List all documents inside a store for that user.
    Uses ONLY local metadata (documents are NOT stored).
    """
    data = load_data()

    if store_name not in data["file_stores"]:
        raise HTTPException(status_code=404, detail="Store not found")

    meta = data["file_stores"][store_name]

    if meta.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Store does not belong to this user")

    documents = meta.get("files", [])

    # Clean response: only useful fields
    docs_clean = []
    for doc in documents:
        docs_clean.append({
            "document_id": doc.get("document_id"),
            "display_name": doc.get("display_name"),
            "size_bytes": doc.get("size_bytes"),
            "uploaded_at": doc.get("uploaded_at"),
            "gemini_indexed": doc.get("gemini_indexed", True)
        })

    return {
        "success": True,
        "user_id": user_id,
        "store": store_name,
        "documents": docs_clean
    }

# ============================================================
# ASK ENDPOINT (NEW FINAL FORMAT)
# ============================================================

@app.get("/{user_id}/{store}", response_class=JSONResponse)
def ask(user_id: str, store: str, ask: str):
    """
    FINAL ASK ENDPOINT
    GET /{user_id}/{store}?ask=QUESTION_TEXT

    - Runs RAG only if read == true
    """
    question = ask
    data = load_data()

    if store not in data["file_stores"]:
        raise HTTPException(404, "Store not found")

    meta = data["file_stores"][store]

    if meta["user_id"] != user_id:
        raise HTTPException(403, "Not your store")

    if not meta.get("read", False):
        raise HTTPException(400, "Store is not live (read=false)")

    api_key = meta["api_key"]
    fs_name = meta["file_search_store_name"]

    client = init_gemini_client(api_key)

    try:
        tool = types.Tool(
            file_search=types.FileSearch(file_search_store_names=[fs_name])
        )

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction="Answer ONLY using information from the documents.",
                tools=[tool],
                temperature=0.2
            )
        )

        text = getattr(resp, "text", "")

        return {"success": True, "answer": text}

    except Exception as e:
        raise HTTPException(500, str(e))
