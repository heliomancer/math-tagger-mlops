import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from math_classifier.train import train
from math_classifier.infer import infer # tbd
from math_classifier.data_local import download_data

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    
    mode = cfg.get("mode", "train")
    
    if mode == "train":
        log.info("Starting Training Pipeline...")
        train(cfg)
        
    elif mode == "download":
        log.info("Downloading Data...")
        download_data()
        
    elif mode == "infer":
        log.info("Starting Inference...")
        infer(cfg)
        
    elif mode == "trt":
        from math_classifier.to_trt import main as trt_main
        trt_main()
        
    else:
        log.error(f"Unknown mode: {mode}. Available: train, download, infer, trt")

if __name__ == "__main__":
    main()
