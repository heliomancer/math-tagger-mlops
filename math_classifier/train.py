from omegaconf import DictConfig

from math_classifier.trainers.catboost_trainer import train_catboost
from math_classifier.trainers.logreg_trainer import train_logreg


def train(cfg: DictConfig):
    model_type = cfg.model.get("type", "logreg")

    if model_type == "logreg":
        print("🚀 Starting PyTorch (LogReg) Training...")
        train_logreg(cfg)

    elif model_type == "catboost":
        print("🐱 Starting CatBoost Training...")
        train_catboost(cfg)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
