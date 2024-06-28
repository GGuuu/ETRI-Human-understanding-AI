import time
import torch

def test(device, test_data, model):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        prediction = None
        for user, date, acc, hr, gps, act in test_data:
            acc = acc.to(device)
            hr = hr.reshape(hr.shape[0], 1, hr.shape[1]).to(device)
            gps = gps.reshape(gps.shape[0], 1, gps.shape[1]).to(device)
            act = act.to(device)


            output = model(acc, hr, act, gps).to(device)
            y_pred = (output > 0.5).float()

            if prediction is None:
                prediction = y_pred
            else:
                prediction = torch.concat([prediction, y_pred])

        end_time = time.time()
        print('Testing Time:', str(end_time - start_time))
    return  prediction