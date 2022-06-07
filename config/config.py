from yacs.config import CfgNode as CN
from datetime import datetime

def get_time():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    return dt_string

def default_cfg():
    cfg = CN()

    cfg.micro = False
    cfg.batch_size = 2400
    cfg.epochs = 2000
    cfg.early_stop = 250

    cfg.loss_fn = "BCEWithLogitsLoss"
    cfg.model = "resnet152"
    cfg.optimizer = "adam"
    cfg.learning_rate = 0.01
    cfg.transform = "compose"
    
    cfg.min_snr = 30
    cfg.max_snr = 30

    cfg.log_file = f"log-{get_time()}"
    cfg.writer_file = f"runs/{cfg.model}-{get_time()}"

    cfg.title = (   "DEEPSIG 2018 Dataset \n" 
                    f"Model:{cfg.model} LossFn:{cfg.loss_fn} Optim: {cfg.optimizer} lr: {cfg.learning_rate} \n" 
                    f"  minSNR:{cfg.max_snr} maxSNR:{cfg.min_snr} \n"
                    "Curriculum Lesson Plan: \n" 
                    "     (30snr - 30snr)   -> "
                    "     (28snr - 30snr)   -> "
                    "     (...snr - 30snr)  -> "
                    "     (-20snr - 30snr)  \n " 
                    f"-----------{get_time()}----------\n")

    return cfg