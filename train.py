import torch

def train(device, train_data, model, optimizer, loss_fn):
    model.train()
    train_loss = 0.0
    train_accuracy = 0
    for _, _, acc, hr, gps, act, label in train_data:
        acc = acc.to(device)
        hr = hr.reshape(hr.shape[0], 1, hr.shape[1]).to(device)
        gps = gps.reshape(gps.shape[0], 1, gps.shape[1]).to(device)
        act = act.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(acc, hr, act, gps)

        loss = loss_fn(output, label)
        y_pred = (output > 0.5).float()
        acc = torch.sum(y_pred == label).item()
        train_accuracy += acc / len(y_pred)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    train_loss = train_loss / len(train_data)
    train_accuracy = train_accuracy / (len(train_data) * 7)

    print('Training Loss:' ,train_loss)
    print('Training ACC:', train_accuracy)
    return train_loss,train_accuracy