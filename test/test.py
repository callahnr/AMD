from tqdm import tqdm
import torch

def test(testloader, criterion, model, device, micro=False):
    
    model.eval()
    running_loss=0
    correct=0
    total=0
    steps = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs,labels = data[0].to(device),data[1].to(device)
            # inputs.shape = batch_size, 2, sequence length (1024)
            # labels.shape = batch_size, num_classes (24)
            # Only one label is set to 1 during training.
            # In Real-World during inference, none of the classes
            # may match.

            outputs = model(inputs)
            # outputs.shape = batch_size, num_classes
            # outputs are logits, no softmax
            
            loss = criterion(outputs,labels)
            running_loss += loss.item()

            total += labels.size(0)
            # Although multiple outputs could be high, the highest is the true label
            # Even though we are training for multiclass in the training data only one
            # answer is ever correct. At inference it is possible that none are correct.
            predicted = (outputs > 0).to(torch.float)

            correct += predicted.eq(labels).all(1).sum().item()
            steps += 1
            if micro:
                break   

    test_loss = running_loss/steps
    accu = 100.*correct/total

    return test_loss, accu 