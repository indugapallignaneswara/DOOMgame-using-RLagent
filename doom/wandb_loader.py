"""W&B artifact loader — download trained models from Weights & Biases."""

import os
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

WANDB_PROJECT = "doom-rl"
WANDB_ENTITY = "ignaneswara-srm-institute-of-science-and-technology"


def list_wandb_artifacts(project=WANDB_PROJECT, entity=WANDB_ENTITY, type_name="model"):
    """List all model artifacts in the W&B project.

    Returns list of dicts with: name, version, aliases, metadata, created_at
    """
    try:
        import wandb
        api = wandb.Api()
        collection_path = f"{entity}/{project}"
        artifacts = []

        for artifact in api.artifacts(type_name=type_name, name=collection_path):
            artifacts.append({
                "name": artifact.name,
                "version": artifact.version,
                "aliases": artifact.aliases,
                "metadata": artifact.metadata or {},
                "created_at": artifact.created_at,
                "size": artifact.size,
                "id": artifact.id,
            })
        return artifacts
    except Exception as e:
        logger.error(f"Failed to list W&B artifacts: {e}")
        return []


def download_wandb_model(artifact_name, version="latest", project=WANDB_PROJECT,
                         entity=WANDB_ENTITY, dest_dir=None):
    """Download a model artifact from W&B and return the local path.

    Args:
        artifact_name: e.g. "ppo_basic_final"
        version: artifact version or "latest"
        project: W&B project name
        entity: W&B entity/team name
        dest_dir: where to save (defaults to config.MODELS_DIR/wandb/)

    Returns:
        dict with: path (local zip path), metadata, artifact_name, scenario, algo
    """
    import wandb
    import config

    if dest_dir is None:
        dest_dir = os.path.join(config.MODELS_DIR, "wandb")
    os.makedirs(dest_dir, exist_ok=True)

    api = wandb.Api()
    artifact_path = f"{entity}/{project}/{artifact_name}:{version}"

    logger.info(f"Downloading W&B artifact: {artifact_path}")
    artifact = api.artifact(artifact_path)
    download_dir = artifact.download(root=dest_dir)

    # Find the .zip file in the downloaded artifact
    zip_path = None
    for fname in os.listdir(download_dir):
        if fname.endswith(".zip"):
            zip_path = os.path.join(download_dir, fname)
            break

    if zip_path is None:
        # Sometimes the artifact downloads to a subdirectory
        for root, dirs, files in os.walk(download_dir):
            for fname in files:
                if fname.endswith(".zip"):
                    zip_path = os.path.join(root, fname)
                    break
            if zip_path:
                break

    if zip_path is None:
        raise FileNotFoundError(f"No .zip model file found in artifact {artifact_name}")

    metadata = artifact.metadata or {}
    scenario = metadata.get("scenario", _infer_scenario(artifact_name))
    algo = metadata.get("algo", _infer_algo(artifact_name))

    return {
        "path": zip_path,
        "metadata": metadata,
        "artifact_name": artifact_name,
        "scenario": scenario,
        "algo": algo.lower() if algo else "ppo",
        "version": artifact.version,
        "created_at": artifact.created_at,
    }


def download_and_register(artifact_name, version="latest", project=WANDB_PROJECT,
                          entity=WANDB_ENTITY, registry_path=None):
    """Download a W&B model and add it to the local registry.

    Returns the registry entry dict.
    """
    import config
    import uuid

    if registry_path is None:
        registry_path = config.REGISTRY_PATH

    result = download_wandb_model(artifact_name, version, project, entity)

    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    # Check if already registered
    for m in registry["models"]:
        if m.get("wandb_artifact") == artifact_name:
            logger.info(f"Artifact {artifact_name} already registered as {m['id']}")
            m["path"] = result["path"]  # Update path in case it moved
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            return m

    # Create registry entry
    entry = {
        "id": str(uuid.uuid4())[:8],
        "name": artifact_name.replace("_final", ""),
        "scenario": result["scenario"],
        "algo": result["algo"],
        "path": result["path"],
        "wandb_artifact": artifact_name,
        "wandb_version": result["version"],
        "hyperparams": {k: v for k, v in result["metadata"].items()
                       if k in ("learning_rate", "n_steps", "clip_range", "gamma", "gae_lambda")},
        "total_timesteps": result["metadata"].get("total_timesteps", 0),
        "created_at": result.get("created_at", datetime.now().isoformat()),
        "mean_reward": result["metadata"].get("mean_reward", 0),
        "total_episodes": 0,
        "source": "wandb",
    }

    registry["models"].append(entry)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Registered W&B model: {entry['id']} ({artifact_name})")
    return entry


def sync_all_wandb_models(project=WANDB_PROJECT, entity=WANDB_ENTITY, registry_path=None):
    """Download and register ALL model artifacts from the W&B project.

    Returns list of registry entries (new + existing).
    """
    artifacts = list_wandb_artifacts(project, entity)
    results = []

    for art in artifacts:
        try:
            entry = download_and_register(
                art["name"].split(":")[0],  # strip version
                version="latest",
                project=project,
                entity=entity,
                registry_path=registry_path,
            )
            results.append(entry)
        except Exception as e:
            logger.error(f"Failed to sync artifact {art['name']}: {e}")

    return results


def _infer_scenario(artifact_name):
    """Infer scenario from artifact name like 'ppo_basic_final'."""
    from doom.scenarios import SCENARIOS
    name_lower = artifact_name.lower()
    for scenario in SCENARIOS:
        if scenario in name_lower:
            return scenario
    return "unknown"


def _infer_algo(artifact_name):
    """Infer algorithm from artifact name."""
    name_lower = artifact_name.lower()
    for algo in ("ppo", "dqn", "a2c"):
        if name_lower.startswith(algo):
            return algo
    return "ppo"
