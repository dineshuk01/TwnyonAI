import json
import asyncio
import os
import shutil
import warnings
from dotenv import load_dotenv

load_dotenv()

# Suppress all warnings to prevent LangChainPendingDeprecationWarning during imports
warnings.filterwarnings("ignore")
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from agent.graph import app as agent_graph
from memory import mongo
import bcrypt

def hash_password(password: str) -> str:
    # Ensure password is not too long for bcrypt
    pwd_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_byte_enc = plain_password.encode('utf-8')[:72]
    hashed_password_byte_enc = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_enc, hashed_password_byte_enc)

app = FastAPI(title="Twnyon AI API")

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    goal: str
    session_id: str
    user_email: str

@app.post("/run")
async def run_agent(request: RunRequest):
    async def event_generator():
        initial_state = {
            "messages": [HumanMessage(content=request.goal)],
            "session_id": request.session_id,
            "user_email": request.user_email,
            "iteration": 0,
            "steps": [],
            "final_answer": None
        }
        
        config = {"configurable": {"thread_id": request.session_id}}
        
        try:
            async for output in agent_graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, state_update in output.items():
                    steps = state_update.get("steps", [])
                    for step in steps:
                        yield json.dumps(step)
                        await asyncio.sleep(0.01)
                    
                    if "final_answer" in state_update and state_update["final_answer"] and "evaluation" in state_update:
                        final_event = {
                            "type": "final",
                            "answer": state_update["final_answer"],
                            "evaluation": state_update["evaluation"]
                        }
                        yield json.dumps(final_event)
                        
            # If the graph finished but evaluation wasn't explicitly yielded:
            # We can fetch the latest state
            latest_state = (await agent_graph.aget_state(config)).values
            if "evaluation" in latest_state and "final_answer" in latest_state:
                final_event = {
                    "type": "final",
                    "answer": latest_state["final_answer"],
                    "evaluation": latest_state["evaluation"]
                }
                yield json.dumps(final_event)

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)})

    return EventSourceResponse(event_generator())

@app.get("/memory/{user_email}")
async def get_memory(user_email: str):
    memories = await mongo.list_memory(user_email)
    return {"memories": memories}

@app.delete("/memory/{user_email}")
async def delete_memory(user_email: str):
    await mongo.clear_memory(user_email)
    return {"status": "cleared"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    messages = await mongo.load_session(session_id)
    return {"messages": messages}

@app.get("/health")
async def health_check():
    return {"status": "ok", "mongo": "connected"}

class TitleRequest(BaseModel):
    message: str

@app.post("/api/generate-title")
async def generate_title(request: TitleRequest):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        prompt = (
            "You are a helpful assistant. Generate a short, concise, and professional title "
            "(maximum of 3 to 4 words) that summarizes the user's message/request. "
            "Do not include quotes, markdown formatting, or punctuation. "
            "Example user message: 'hi', 'hello', 'hey' -> Title: 'Greeting Exchange'. "
            "Example user message: 'can you write a python script to parse CSV files' -> Title: 'CSV Parsing Script'."
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=request.message)
        ]
        response = await llm.ainvoke(messages)
        title = response.content.strip().strip('"').strip("'")
        return {"title": title}
    except Exception as e:
        fallback_title = request.message[:30] + ("..." if len(request.message) > 30 else "")
        return {"title": fallback_title}

@app.post("/upload-file")
async def upload_file(user_email: str, file: UploadFile = File(...)):
    user_dir = os.path.join("uploads", user_email)
    os.makedirs(user_dir, exist_ok=True)
    file_location = os.path.join(user_dir, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    return {"url": f"{backend_url}/uploads/{user_email}/{file.filename}", "filename": file.filename}

@app.get("/workspace-files")
async def get_workspace_files(user_email: str):
    files = []
    user_dir = os.path.join("uploads", user_email)
    if os.path.exists(user_dir):
        files = [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))]
    return {"files": files}

@app.delete("/workspace-files/{filename}")
async def delete_workspace_file(filename: str, user_email: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("uploads", user_email, safe_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    return {"status": "not_found", "error": "File not found"}

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class SigninRequest(BaseModel):
    email: str
    password: str

@app.post("/api/signup")
async def signup(request: SignupRequest):
    existing_user = await mongo.get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = hash_password(request.password)
    await mongo.create_user(request.name, request.email, hashed_password)
    return {"message": "User created successfully"}

@app.post("/api/signin")
async def signin(request: SigninRequest):
    user = await mongo.get_user_by_email(request.email)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    
    return {
        "message": "Signin successful",
        "user": {
            "name": user.get("name"),
            "email": user.get("email")
        }
    }
