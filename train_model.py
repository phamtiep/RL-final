

device = 'cuda'
def train_1_epoch(num, dataloader, model, optimizer, lr, loss_fn):

    debug = False
    if (num%100 == 0): debug = True
    total_loss = 0
    for id, (X, y) in enumerate(dataloader):
        model.train()
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss / len(dataloader)
    if (debug == True):
        print(f'Epoch number {id} loss : {total_loss}')
    
    return total_loss

def train_model(num_epoch, dataloader, model, optimzer, lr, loss_fn):
    current_loss = 100
    while(current_loss > 0.001):
        for epoch in range(1, num_epoch + 1):
            current_loss = train_1_epoch(epoch, dataloader, model, optimzer, lr, loss_fn)
        