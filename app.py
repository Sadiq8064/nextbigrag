# gfsapi.py
"""
Full backend with:
- Firebase Firestore user system (email, password, API keys)
- Multi-store per user (display_name, store_id)
- Per-API-key 1GB quota (calculated from Firestore file sizes)
- read (boolean) to control RAG accessibility
- Make-Live returning only final Ask-URL string
- Ask endpoint using /{user_id}/{store_id}?ask=...
- Temporary upload deletion (documents never stored permanently on disk)
- All metadata (stores, files, usage) in Firestore (NO local JSON)
"""

import os
import time
import shutil
import re
import uuid
import requests
from pathlib import Path
from typing import List

import aiofiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

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


# ---------------- Helpers ----------------

def ensure_dirs():
    """Ensure only upload temp directory exists."""
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


ensure_dirs()


def verify_gemini_key(api_key: str) -> bool:
    """Verify Gemini API key by doing a tiny generate_content call."""
    try:
        if genai is None or types is None:
            return False

        client = genai.Client(api_key=api_key)

        _ = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )

        return True

    except Exception:
        # Any failure: treat as invalid
        return False


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
    """Fallback: list documents using REST if SDK doesn't give resource name."""
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


# ---------------- Firestore Quota Helpers ----------------

def compute_api_key_usage_bytes(user_id: str, api_key: str) -> int:
    """
    Compute total bytes used for a given (user_id, api_key) pair,
    based ONLY on Firestore metadata:

    users/{user_id}/stores/{store_id}
        api_key == api_key
        users/{user_id}/stores/{store_id}/files/{doc_id}
            size_bytes
    """
    total = 0
    stores_ref = db.collection("users").document(user_id).collection("stores")
    # Only stores that use this api_key
    store_docs = stores_ref.where("api_key", "==", api_key).stream()

    for store_doc in store_docs:
        files_ref = store_doc.reference.collection("files")
        for file_doc in files_ref.stream():
            data = file_doc.to_dict() or {}
            size = int(data.get("size_bytes", 0) or 0)
            total += size

    return total


def pick_api_key_for_new_store(user_id: str) -> str:
    """
    Pick an API key (among user's apiKeys) that has remaining quota < 1GB.
    """
    doc = db.collection("users").document(user_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")

    api_keys = doc.to_dict().get("apiKeys", [])
    if not api_keys:
        raise HTTPException(status_code=400, detail="User has no API keys")

    for entry in api_keys:
        key = entry.get("key")
        used = compute_api_key_usage_bytes(user_id, key)
        if used < MAX_TOTAL_BYTES_PER_API_KEY:
            return key

    raise HTTPException(status_code=400, detail="All API keys reached 1GB limit.")


# ============================================================
# Request Models
# ============================================================

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


# ============================================================
# USERS
# ============================================================

@app.post("/users/register")
def register_user(payload: RegisterRequest):
    existing = list(
        db.collection("users")
        .where("email", "==", payload.email)
        .limit(1)
        .stream()
    )
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
    docs = list(
        db.collection("users")
        .where("email", "==", payload.email)
        .limit(1)
        .stream()
    )
    if not docs:
        return JSONResponse({"error": "Invalid email or password"}, 401)

    doc_data = docs[0].to_dict()
    if doc_data.get("password") != payload.password:
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

    if not isinstance(keys, list):
        keys = []

    new_keys = [k for k in keys if k.get("key") != payload.api_key]

    if len(new_keys) == len(keys):
        return JSONResponse({"error": "API key not found"}, 404)

    ref.update({"apiKeys": new_keys})

    return {"success": True, "apiKeyCount": len(new_keys)}


# ============================================================
# STORE CREATE (Firestore)
# ============================================================

@app.post("/stores/create")
def create_store(payload: CreateStoreRequest):
    """
    Create a new File Search store for a user.

    Firestore path:
      users/{user_id}/stores/{store_id}
    """
    user_id = payload.user_id

    api_key = pick_api_key_for_new_store(user_id)
    client = init_gemini_client(api_key)

    suffix = uuid.uuid4().hex[:8]
    store_id = f"{clean_filename(payload.display_name)}_{suffix}"

    # Create Gemini File Search store
    try:
        fs_store = client.file_search_stores.create(
            config={"display_name": payload.display_name}
        )
        fs_name = getattr(fs_store, "name", None) or fs_store
    except Exception as e:
        raise HTTPException(500, f"Gemini store create failed: {e}")

    # Save store metadata in Firestore
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    store_ref.set({
        "store_id": store_id,
        "display_name": payload.display_name,
        "file_search_store_name": fs_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "api_key": api_key,
        "read": False
    })

    return {
        "success": True,
        "store_id": store_id,
        "display_name": payload.display_name
    }


# ============================================================
# TOGGLE READ FLAG (Firestore)
# ============================================================

@app.post("/stores/{user_id}/{store_id}/toggle-read")
def toggle_read(user_id: str, store_id: str):
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    doc = store_ref.get()

    if not doc.exists:
        raise HTTPException(404, "Store not found")

    meta = doc.to_dict() or {}
    current_read = bool(meta.get("read", False))

    store_ref.update({"read": not current_read})

    return {"success": True, "read": not current_read}


# ============================================================
# MAKE-LIVE → RETURNS ONLY FINAL URL STRING (Firestore)
# ============================================================

@app.get("/stores/{user_id}/{store_id}/make-live", response_class=PlainTextResponse)
def make_live(user_id: str, store_id: str):
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    doc = store_ref.get()

    if not doc.exists:
        raise HTTPException(404, "Store not found")

    meta = doc.to_dict() or {}

    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Store does not belong to this user")

    if not meta.get("read", False):
        raise HTTPException(400, "Store is not live. Toggle read first.")

    if not BASE_URL:
        raise HTTPException(500, "BASE_URL not set in environment")

    # REQUIRED FORMAT:
    # https://BASE_URL/{user_id}/{store_id}?ask={question}
    final_url = f"{BASE_URL}/{user_id}/{store_id}?ask={{question}}"

    # Return EXACT plain text URL
    return final_url


# ============================================================
# UPLOAD DOCUMENTS (Firestore, temp-only disk)
# ============================================================

@app.post("/stores/{user_id}/{store_id}/upload")
async def upload_docs(
    user_id: str,
    store_id: str,
    limit: bool = Form(True),
    files: List[UploadFile] = File(...)
):
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    store_doc = store_ref.get()

    if not store_doc.exists:
        raise HTTPException(404, "Store not found")

    meta = store_doc.to_dict() or {}

    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Not your store")

    fs_name = meta.get("file_search_store_name")
    api_key = meta.get("api_key")

    client = init_gemini_client(api_key)

    temp_folder = UPLOAD_ROOT / store_id
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
                        if temp_path.exists():
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

        # If size loop broke due to limit, skip further logic
        if size > MAX_FILE_BYTES and limit:
            continue

        # Check 1GB quota
        if size + current_usage > MAX_TOTAL_BYTES_PER_API_KEY:
            if temp_path.exists():
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
            except Exception:
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
            if temp_path.exists():
                os.remove(temp_path)
            continue

        # delete temp after successful upload
        if temp_path.exists():
            os.remove(temp_path)

        # Save file metadata in Firestore
        if document_id:
            file_ref = store_ref.collection("files").document(document_id)
            file_ref.set({
                "document_id": document_id,
                "display_name": filename,
                "size_bytes": size,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "document_resource": document_resource
            })

        current_usage += size
        results.append({
            "filename": filename,
            "uploaded": True,
            "indexed": True,
            "document_id": document_id
        })

    return {"success": True, "results": results}


# ============================================================
# DELETE DOCUMENT (Firestore)
# ============================================================

@app.delete("/stores/{user_id}/{store_id}/documents/{doc_id}")
def delete_doc(user_id: str, store_id: str, doc_id: str):
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    store_doc = store_ref.get()

    if not store_doc.exists:
        raise HTTPException(404, "Store not found")

    meta = store_doc.to_dict() or {}
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Not your store")

    fs_name = meta.get("file_search_store_name")
    api_key = meta.get("api_key")

    # Delete from Gemini
    url = f"{GEMINI_REST_BASE}/{fs_name}/documents/{doc_id}"
    resp = requests.delete(url, params={"force": "true", "key": api_key})

    if resp.status_code not in (200, 204):
        return JSONResponse({"error": resp.text}, resp.status_code)

    # Delete file metadata from Firestore
    store_ref.collection("files").document(doc_id).delete()

    return {"success": True}


# ============================================================
# DELETE STORE (Firestore)
# ============================================================

@app.delete("/stores/{user_id}/{store_id}")
def delete_store(user_id: str, store_id: str):
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    store_doc = store_ref.get()

    if not store_doc.exists:
        raise HTTPException(404, "Store not found")

    meta = store_doc.to_dict() or {}
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Not your store")

    fs_name = meta.get("file_search_store_name")
    api_key = meta.get("api_key")

    # Delete Gemini store
    try:
        client = init_gemini_client(api_key)
        client.file_search_stores.delete(name=fs_name, config={"force": True})
    except Exception:
        pass

    # Delete all file metadata
    files_ref = store_ref.collection("files")
    for fdoc in files_ref.stream():
        fdoc.reference.delete()

    # Delete store document
    store_ref.delete()

    # delete temp upload folder for this store, if any
    folder = UPLOAD_ROOT / store_id
    if folder.exists():
        shutil.rmtree(folder)

    return {"success": True}


# ============================================================
# LIST ALL DOCUMENTS IN A STORE (Firestore)
# ============================================================

@app.get("/stores/{user_id}/{store_id}/documents")
def list_documents(user_id: str, store_id: str):
    """
    List all documents inside a store for that user.
    Uses ONLY Firestore metadata (documents are NOT stored locally).
    """
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    store_doc = store_ref.get()

    if not store_doc.exists:
        raise HTTPException(status_code=404, detail="Store not found")

    meta = store_doc.to_dict() or {}

    if meta.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Store does not belong to this user")

    docs_clean = []
    for fdoc in store_ref.collection("files").stream():
        data = fdoc.to_dict() or {}
        docs_clean.append({
            "document_id": data.get("document_id"),
            "display_name": data.get("display_name"),
            "size_bytes": data.get("size_bytes"),
            "uploaded_at": data.get("uploaded_at"),
            "gemini_indexed": True  # if it's here, we assume it's indexed
        })

    return {
        "success": True,
        "user_id": user_id,
        "store": store_id,
        "documents": docs_clean
    }


# ============================================================
# USER USAGE ANALYTICS (API KEYS + STORES + DOCUMENTS) - Firestore
# ============================================================

@app.get("/users/{user_id}/usage")
def get_usage(user_id: str):
    """
    Returns usage summary for the user:
      - API key usage (bytes used + remaining)
      - Store usage per store
      - File list with sizes

    All values are derived from Firestore:

    users/{user_id}
        apiKeys: [ { key } ]
    users/{user_id}/stores/{store_id}
        api_key, display_name
    users/{user_id}/stores/{store_id}/files/{doc_id}
        size_bytes, etc.
    """
    # Get user data
    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        raise HTTPException(404, "User not found")

    user = user_doc.to_dict() or {}
    api_keys = user.get("apiKeys", [])

    # ---------- BUILD API KEY USAGE ----------
    key_usage = []
    for entry in api_keys:
        key = entry.get("key")
        used = compute_api_key_usage_bytes(user_id, key)
        remaining = MAX_TOTAL_BYTES_PER_API_KEY - used

        key_usage.append({
            "api_key": key,
            "used_bytes": used,
            "remaining_bytes": max(0, remaining)
        })

    # ---------- BUILD STORE USAGE ----------
    store_usage = []
    stores_ref = db.collection("users").document(user_id).collection("stores")
    for store_doc in stores_ref.stream():
        meta = store_doc.to_dict() or {}

        if meta.get("user_id") != user_id:
            continue

        files_ref = store_doc.reference.collection("files")
        files_list = []
        total_size = 0
        for fdoc in files_ref.stream():
            fdata = fdoc.to_dict() or {}
            s = int(fdata.get("size_bytes", 0) or 0)
            total_size += s

            files_list.append({
                "document_id": fdata.get("document_id"),
                "display_name": fdata.get("display_name"),
                "size_bytes": s,
                "uploaded_at": fdata.get("uploaded_at")
            })

        store_usage.append({
            "store_id": meta.get("store_id"),
            "display_name": meta.get("display_name"),
            "api_key_used": meta.get("api_key"),
            "total_store_bytes": total_size,
            "document_count": len(files_list),
            "read": meta.get("read", False),
            "created_at": meta.get("created_at"),
            "files": files_list
        })

    return {
        "success": True,
        "user_id": user_id,
        "api_key_usage": key_usage,
        "stores": store_usage
    }



@app.get("/users/{user_id}/stores")
def list_stores(user_id: str):
    """
    Returns a list of all stores owned by the user.
    
    Firestore path:
      users/{user_id}/stores/{store_id}
    """
    # Check user exists
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        raise HTTPException(404, "User not found")

    stores_ref = user_ref.collection("stores")

    stores_list = []
    for doc in stores_ref.stream():
        data = doc.to_dict() or {}
        stores_list.append({
            "store_id": data.get("store_id"),
            "display_name": data.get("display_name"),
            "read": data.get("read", False),
            "created_at": data.get("created_at"),
            "api_key_used": data.get("api_key"),
            "file_search_store_name": data.get("file_search_store_name")
        })

    return {
        "success": True,
        "user_id": user_id,
        "store_count": len(stores_list),
        "stores": stores_list
    }
# ============================================================
# ASK ENDPOINT (FINAL FORMAT WITH CITATIONS, Firestore)
# ============================================================
@app.get("/{user_id}/{store_id}", response_class=JSONResponse)
def ask(user_id: str, store_id: str, ask: str):
    """
    FINAL ASK ENDPOINT
    GET /{user_id}/{store_id}?ask=QUESTION_TEXT

    - Runs RAG only if read == true
    - Returns answer + citations (grounding metadata)
    """
    question = ask

    # Fetch store metadata from Firestore
    store_ref = db.collection("users").document(user_id).collection("stores").document(store_id)
    store_doc = store_ref.get()

    if not store_doc.exists:
        raise HTTPException(404, "Store not found")

    meta = store_doc.to_dict() or {}

    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Not your store")

    if not meta.get("read", False):
        raise HTTPException(400, "Store is not live (read=false)")

    api_key = meta.get("api_key")
    fs_name = meta.get("file_search_store_name")

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

        answer = getattr(resp, "text", "")

        # ----- Extract citations / grounding metadata -----
        grounding = None
        if hasattr(resp, "candidates") and len(resp.candidates) > 0:
            grounding = getattr(resp.candidates[0], "grounding_metadata", None)

        return {
            "success": True,
            "answer": answer,
            "citations": grounding
        }

    except Exception as e:
        raise HTTPException(500, str(e))
