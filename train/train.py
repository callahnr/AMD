from sched import scheduler
from tqdm import tqdm

def train(trainloader, optimizer, criterion, model, device, micro=False):
    classes = [ "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", 
                "32PSK", "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", 
                "32QAM", "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", 
                "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]
    model.train()
    running_loss=0
    correct=0
    total=0
    steps = 0


    
    for inputs, labels in tqdm(trainloader):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        steps += 1
        correct += predicted.eq(labels.argmax(1)).sum().item()
        if micro:
            break
    train_loss=running_loss/steps
    accu=100.*correct/total


    return train_loss, accu

