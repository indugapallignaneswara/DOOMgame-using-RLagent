"""DOOM RL Web Application - Flask + SocketIO."""

import os
import json
import logging
import threading
import uuid
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY

socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Late imports to avoid circular deps - initialized after app starts
trainer = None
player = None
mindmap_player = None

def get_trainer():
    global trainer
    if trainer is None:
        from doom.trainer import TrainingManager
        trainer = TrainingManager(socketio)
    return trainer

def get_player():
    global player
    if player is None:
        from doom.player import AIPlayer
        player = AIPlayer(socketio)
    return player

def get_mindmap():
    global mindmap_player
    if mindmap_player is None:
        from doom.mindmap import MindMapPlayer
        mindmap_player = MindMapPlayer(socketio)
    return mindmap_player

# ─── Model Registry Helpers ───

registry_lock = threading.Lock()
history_lock = threading.Lock()

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "models", "training_history.json")
USERS_PATH = os.path.join(os.path.dirname(__file__), "models", "users.json")

def load_registry():
    with registry_lock:
        if os.path.exists(config.REGISTRY_PATH):
            with open(config.REGISTRY_PATH, "r") as f:
                return json.load(f)
        return {"models": []}

def save_registry(data):
    with registry_lock:
        os.makedirs(os.path.dirname(config.REGISTRY_PATH), exist_ok=True)
        with open(config.REGISTRY_PATH, "w") as f:
            json.dump(data, f, indent=2)

def load_history():
    with history_lock:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
        return {"sessions": []}

def save_history(data):
    with history_lock:
        with open(HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=2)

def save_training_session(session_data):
    """Append or update a training session in history."""
    history = load_history()
    # Update existing or append
    found = False
    for i, s in enumerate(history["sessions"]):
        if s["session_id"] == session_data["session_id"]:
            history["sessions"][i] = session_data
            found = True
            break
    if not found:
        history["sessions"].insert(0, session_data)
    # Keep last 100 sessions
    history["sessions"] = history["sessions"][:100]
    save_history(history)

def load_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r") as f:
            return json.load(f)
    return {"users": []}

def save_users(data):
    with open(USERS_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ─── HTTP Routes ───

@app.route("/")
def index():
    registry = load_registry()
    from doom.scenarios import list_scenarios
    return render_template("index.html",
                           model_count=len(registry["models"]),
                           scenario_count=len(list_scenarios()))

@app.route("/train")
def train_page():
    from doom.scenarios import SCENARIOS, list_scenarios
    first_key = list_scenarios()[0] if list_scenarios() else "basic"
    first = SCENARIOS.get(first_key, {})
    hp = first.get("default_hyperparams", {})
    rw = first.get("reward_defaults", {})
    defaults = {**hp, **rw, "scenario": first_key}
    return render_template("train.html", defaults=defaults)

@app.route("/play")
def play_page():
    return render_template("play.html")

@app.route("/models")
def models_page():
    return render_template("models.html")

@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluate.html")

@app.route("/mindmap")
def mindmap_page():
    return render_template("mindmap.html")

# ─── API Routes ───

@app.route("/api/scenarios")
def api_scenarios():
    from doom.scenarios import SCENARIOS
    scenarios_list = []
    for key, val in SCENARIOS.items():
        entry = {
            "id": key,
            "name": key.replace("_", " ").title(),
            "description": val.get("description", ""),
            "num_actions": val.get("num_actions", 0),
            "defaults": {},
        }
        # Merge hyperparams and reward defaults into a single defaults dict
        if "default_hyperparams" in val:
            entry["defaults"].update(val["default_hyperparams"])
        if "reward_defaults" in val:
            entry["defaults"].update(val["reward_defaults"])
        scenarios_list.append(entry)
    return jsonify({"scenarios": scenarios_list})

@app.route("/api/models")
def api_models():
    registry = load_registry()
    return jsonify({"models": registry["models"]})

@app.route("/api/models/<model_id>")
def api_model_detail(model_id):
    registry = load_registry()
    for m in registry["models"]:
        if m["id"] == model_id:
            return jsonify(m)
    return jsonify({"error": "Model not found"}), 404

@app.route("/api/models/<model_id>", methods=["DELETE"])
def api_delete_model(model_id):
    registry = load_registry()
    model = None
    for m in registry["models"]:
        if m["id"] == model_id:
            model = m
            break
    if not model:
        return jsonify({"error": "Model not found"}), 404

    # Remove model file if it exists
    if "path" in model and os.path.exists(model["path"]):
        try:
            os.remove(model["path"])
        except OSError:
            pass

    registry["models"] = [m for m in registry["models"] if m["id"] != model_id]
    save_registry(registry)
    return jsonify({"success": True})

@app.route("/api/training/status")
def api_training_status():
    t = get_trainer()
    statuses = {}
    for sid, info in t.active_sessions.items():
        statuses[sid] = info.get("status", "unknown")
    return jsonify(statuses)

# ─── Training History API ───

@app.route("/api/training/history")
def api_training_history():
    history = load_history()
    user = request.args.get("user")
    if user:
        history["sessions"] = [s for s in history["sessions"] if s.get("user") == user]
    return jsonify(history)

@app.route("/api/training/history/<session_id>")
def api_training_session(session_id):
    history = load_history()
    for s in history["sessions"]:
        if s["session_id"] == session_id:
            return jsonify(s)
    return jsonify({"error": "Session not found"}), 404

# ─── User Profile API ───

@app.route("/api/users/register", methods=["POST"])
def api_register_user():
    data = request.get_json()
    username = data.get("username", "").strip()
    if not username or len(username) < 2:
        return jsonify({"error": "Username must be at least 2 characters"}), 400

    users = load_users()
    # Check if username taken
    for u in users["users"]:
        if u["username"].lower() == username.lower():
            return jsonify({"error": "Username already taken"}), 409

    from datetime import datetime
    user = {
        "id": str(uuid.uuid4())[:8],
        "username": username,
        "created_at": datetime.now().isoformat(),
        "models_trained": 0,
        "total_timesteps": 0,
    }
    users["users"].append(user)
    save_users(users)
    return jsonify(user), 201

@app.route("/api/users/login", methods=["POST"])
def api_login_user():
    data = request.get_json()
    username = data.get("username", "").strip()
    users = load_users()
    for u in users["users"]:
        if u["username"].lower() == username.lower():
            return jsonify(u)
    return jsonify({"error": "User not found"}), 404

@app.route("/api/users/<user_id>/stats")
def api_user_stats(user_id):
    users = load_users()
    for u in users["users"]:
        if u["id"] == user_id:
            # Count models and sessions
            registry = load_registry()
            history = load_history()
            user_models = [m for m in registry["models"] if m.get("user") == u["username"]]
            user_sessions = [s for s in history["sessions"] if s.get("user") == u["username"]]
            return jsonify({
                **u,
                "models_count": len(user_models),
                "sessions_count": len(user_sessions),
            })
    return jsonify({"error": "User not found"}), 404

# ─── SocketIO Events ───

@socketio.on("connect")
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on("start_training")
def handle_start_training(data):
    try:
        from datetime import datetime
        session_id = str(uuid.uuid4())[:8]
        scenario_key = data.get("scenario", "basic")
        total_timesteps = int(data.get("total_timesteps", 100000))
        username = data.get("user", "anonymous")

        hyperparams = {
            "learning_rate": float(data.get("learning_rate", 0.0001)),
            "n_steps": int(data.get("n_steps", 2048)),
            "clip_range": float(data.get("clip_range", 0.2)),
            "gamma": float(data.get("gamma", 0.99)),
            "gae_lambda": float(data.get("gae_lambda", 0.95)),
        }

        reward_config = {
            "kill_reward": float(data.get("kill_reward", 0)),
            "miss_penalty": float(data.get("miss_penalty", 0)),
            "step_penalty": float(data.get("step_penalty", 0)),
            "damage_penalty": float(data.get("damage_penalty", 0)),
            "ammo_penalty": float(data.get("ammo_penalty", 0)),
        }

        # Save session to history
        save_training_session({
            "session_id": session_id,
            "user": username,
            "scenario": scenario_key,
            "status": "started",
            "total_timesteps": total_timesteps,
            "hyperparams": hyperparams,
            "reward_config": reward_config,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "mean_reward": None,
            "total_episodes": 0,
        })

        # Emit "started" FIRST so the frontend creates the tab immediately
        emit("training_status", {"session_id": session_id, "status": "started", "scenario": scenario_key})

        t = get_trainer()
        t.start_training(session_id, scenario_key, hyperparams, reward_config, total_timesteps)
        logger.info(f"Training started: {session_id} on {scenario_key} by {username}")

    except Exception as e:
        logger.error(f"Training start failed: {e}")
        emit("training_status", {"session_id": "", "status": "failed", "error": str(e)})

@socketio.on("pause_training")
def handle_pause_training(data):
    session_id = data.get("session_id")
    if session_id:
        get_trainer().pause_training(session_id)
        emit("training_status", {"session_id": session_id, "status": "paused"})

@socketio.on("resume_training")
def handle_resume_training(data):
    session_id = data.get("session_id")
    if session_id:
        get_trainer().resume_training(session_id)
        emit("training_status", {"session_id": session_id, "status": "resumed"})

@socketio.on("stop_training")
def handle_stop_training(data):
    session_id = data.get("session_id")
    if session_id:
        get_trainer().stop_training(session_id)
        emit("training_status", {"session_id": session_id, "status": "stopped"})

@socketio.on("start_demo")
def handle_start_demo(data):
    try:
        session_id = str(uuid.uuid4())[:8]
        model_path = data.get("model_path", "")
        scenario_key = data.get("scenario", "basic")
        speed = float(data.get("speed", 0.05))
        saliency_mode = data.get("saliency_mode", "off")
        if saliency_mode not in ("off", "policy", "value"):
            saliency_mode = "off"
        prediction_mode = bool(data.get("prediction_mode", False))
        prediction_every = int(data.get("prediction_every", 8))

        model_id = data.get("model_id")
        if model_id and not model_path:
            registry = load_registry()
            for m in registry["models"]:
                if m["id"] == model_id:
                    model_path = m["path"]
                    scenario_key = m.get("scenario", scenario_key)
                    break

        if not model_path or not os.path.exists(model_path):
            emit("demo_status", {"session_id": session_id, "status": "failed", "error": "Model not found"})
            return

        p = get_player()
        p.start_demo(session_id, model_path, scenario_key, speed,
                     saliency_mode=saliency_mode,
                     prediction_mode=prediction_mode,
                     prediction_every=prediction_every)
        emit("demo_status", {"session_id": session_id, "status": "started"})

    except Exception as e:
        logger.error(f"Demo start failed: {e}")
        emit("demo_status", {"session_id": "", "status": "failed", "error": str(e)})


@socketio.on("set_saliency_mode")
def handle_set_saliency_mode(data):
    """Toggle saliency overlay mid-demo (off / policy / value)."""
    session_id = data.get("session_id")
    mode = data.get("mode", "off")
    if mode not in ("off", "policy", "value"):
        return
    if session_id:
        get_player().set_saliency_mode(session_id, mode)


@socketio.on("set_prediction_mode")
def handle_set_prediction_mode(data):
    session_id = data.get("session_id")
    enabled = bool(data.get("enabled", False))
    every = data.get("every")
    if session_id:
        get_player().set_prediction_mode(session_id, enabled,
                                         every=int(every) if every else None)


@socketio.on("prediction_response")
def handle_prediction_response(data):
    session_id = data.get("session_id")
    choice = data.get("choice")
    if session_id is None:
        return
    get_player().submit_prediction(session_id, choice)

@socketio.on("stop_demo")
def handle_stop_demo(data):
    session_id = data.get("session_id")
    if session_id:
        get_player().stop_demo(session_id)
        emit("demo_status", {"session_id": session_id, "status": "stopped"})

@socketio.on("start_mindmap")
def handle_start_mindmap(data):
    try:
        session_id = str(uuid.uuid4())[:8]
        model_id = data.get("model_id", "")
        scenario_key = data.get("scenario", "basic")
        speed = float(data.get("speed", 0.05))

        model_path = ""
        if model_id:
            registry = load_registry()
            for m in registry["models"]:
                if m["id"] == model_id:
                    model_path = m["path"]
                    scenario_key = m.get("scenario", scenario_key)
                    break

        if not model_path or not os.path.exists(model_path):
            emit("mindmap_status", {"session_id": session_id, "status": "error", "error": "Model not found"})
            return

        mm = get_mindmap()
        mm.start(session_id, model_path, scenario_key, speed)
        emit("mindmap_status", {"session_id": session_id, "status": "started"})

    except Exception as e:
        logger.error(f"Mind Map start failed: {e}")
        emit("mindmap_status", {"session_id": "", "status": "error", "error": str(e)})

@socketio.on("stop_mindmap")
def handle_stop_mindmap(data):
    session_id = data.get("session_id")
    if session_id:
        get_mindmap().stop(session_id)

@socketio.on("set_demo_speed")
def handle_set_demo_speed(data):
    speed = float(data.get("speed", 0.05))
    # Update speed on all active demos for this client
    p = get_player()
    for sid in list(p.active_demos.keys()):
        p.set_demo_speed(sid, speed)

@socketio.on("start_evaluation")
def handle_start_evaluation(data):
    try:
        model_id = data.get("model_id")
        scenario_key = data.get("scenario", "basic")
        n_episodes = int(data.get("n_episodes", 10))

        # Look up model
        model_path = data.get("model_path", "")
        if model_id and not model_path:
            registry = load_registry()
            for m in registry["models"]:
                if m["id"] == model_id:
                    model_path = m["path"]
                    scenario_key = m.get("scenario", scenario_key)
                    break

        if not model_path or not os.path.exists(model_path):
            emit("evaluation_result", {"error": "Model not found"})
            return

        def run_eval():
            try:
                from doom.evaluator import Evaluator
                evaluator = Evaluator()
                results = evaluator.evaluate(model_path, scenario_key, n_episodes)
                socketio.emit("evaluation_result", results)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                socketio.emit("evaluation_result", {"error": str(e)})

        thread = threading.Thread(target=run_eval, daemon=True)
        thread.start()
        emit("evaluation_status", {"status": "running", "n_episodes": n_episodes})

    except Exception as e:
        logger.error(f"Evaluation start failed: {e}")
        emit("evaluation_result", {"error": str(e)})

# ─── Main ───

if __name__ == "__main__":
    logger.info(f"Starting DOOM RL Web App on {config.HOST}:{config.PORT}")
    logger.info(f"Scenarios path: {config.SCENARIOS_PATH}")
    logger.info(f"Models directory: {config.MODELS_DIR}")
    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG, allow_unsafe_werkzeug=True)
