from datetime import datetime
import os

def setup_logs(cfg):
    '''
    setip_logs: Creates folders with 
    timestamps for each sessions logs, 
    runs(for tensorboard), and checkpoints.
    -   cfg = The configuration object
    '''
    now = datetime.now() 
    dt_string = now.strftime("%d-%m-%Y__%H-%M-%S")

    save_dir = f'./logs/{cfg.model}/{dt_string}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    log_file = save_dir + cfg.log_file
    with open(log_file, 'a') as f:
            f.writelines(cfg.title)

    chkpt_dir = f'checkpoints/{cfg.model}-{dt_string}/'
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
        
    return save_dir, log_file, chkpt_dir 


def log_results(cfg, optimizer, train_losses, train_accu, 
                eval_losses, eval_accu, 
                epoch, writer, log_file,
                lesson, min_snr, max_acc,
                best_epoch, epoch_since_best):

    writer.add_scalar('Loss/train', train_losses, epoch)
    writer.add_scalar('Loss/eval', eval_losses, epoch)
    writer.add_scalar('Accuracy/train', train_accu, epoch)
    writer.add_scalar('Accuracy/eval', eval_accu, epoch)

   
    
    now = datetime.now() 
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    
    output = (  f'Lesson: {lesson+1}, Epoch: {epoch}, ' 
                f'minSNR:{min_snr}, maxSNR: 30,  \n'
                f'LR = {optimizer.state_dict()["param_groups"][0]["lr"]} \n'
                f'Train Loss: {train_losses:.3f} | Accuracy: {train_accu:.3f} \n'
                f'Test Loss: {eval_losses:.3f} | Accuracy: {eval_accu:.3f} \n'
                f'------------------------------------------------------------------------ \n'
                f'Best Epoch: {best_epoch} | Best Accuracy: {max_acc:.3f} | {epoch_since_best} epochs since improvement. \n \n'
                f'--------------------------- {dt_string} -------------------------------- \n')

    print(output)

    with open(log_file, 'a') as f:
        f.writelines(output)


def log_lesson(lesson, log_file):
    log_out = ( f"Lesson {lesson} complete \n \n"
                f'--------------------------- {datetime.today()} -------------------------------- \n')
    print(log_out)

    with open(log_file, 'a') as f:
            f.writelines(log_out)

def log_header(cfg, log_file):
    with open(log_file, 'a') as f:
            f.writelines(cfg.dump())
    return cfg.dump()
