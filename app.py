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

# ─── Model Registry Helpers ───

registry_lock = threading.Lock()

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
    return render_template("train.html")

@app.route("/play")
def play_page():
    return render_template("play.html")

@app.route("/models")
def models_page():
    return render_template("models.html")

@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluate.html")

# ─── API Routes ───

@app.route("/api/scenarios")
def api_scenarios():
    from doom.scenarios import SCENARIOS
    return jsonify(SCENARIOS)

@app.route("/api/models")
def api_models():
    registry = load_registry()
    return jsonify(registry["models"])

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
        session_id = str(uuid.uuid4())[:8]
        scenario_key = data.get("scenario", "basic")
        total_timesteps = int(data.get("total_timesteps", 100000))

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

        t = get_trainer()
        t.start_training(session_id, scenario_key, hyperparams, reward_config, total_timesteps)

        emit("training_status", {"session_id": session_id, "status": "started", "scenario": scenario_key})
        logger.info(f"Training started: {session_id} on {scenario_key}")

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

        # Look up model path from registry if model_id given
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
        p.start_demo(session_id, model_path, scenario_key, speed)
        emit("demo_status", {"session_id": session_id, "status": "started"})

    except Exception as e:
        logger.error(f"Demo start failed: {e}")
        emit("demo_status", {"session_id": "", "status": "failed", "error": str(e)})

@socketio.on("stop_demo")
def handle_stop_demo(data):
    session_id = data.get("session_id")
    if session_id:
        get_player().stop_demo(session_id)
        emit("demo_status", {"session_id": session_id, "status": "stopped"})

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
