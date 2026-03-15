import os
import time
from ANN import *
from ERNN import *
import numpy as np
from CERNN import *
from tqdm import tqdm
from torch import optim
from itertools import product
from californiahousing import *
from torch.utils.data import DataLoader

# change K, distribution and data_generator every time before start
param_config = {
    'K': 5,
    'epochs': 400,
    'model': 'ERNN',
    'batch_size': 648,
    'learning_rate': 1e-4,
    'expectile_level': 0.2,
    'hidden_size_range': list(range(4, 7)),
    'l2_lambda_range': np.arange(0.0001, 0.0011, 0.0001),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if param_config['model'] == 'ANN':
    param_config['l2_lambda_range'] = [0]

if not os.path.exists('model'):
    os.mkdir('model')
if param_config['model'] == 'CERNN':
    param_config['model_save_path'] = 'model/{}_K={}_hiddensize={}_λ={}_mae={}_rmse={}.pth'
elif param_config['model'] in ['ERNN', 'ANN']:
    param_config['model_save_path'] = 'model/{}_τ={}_hiddensize={}_λ={}_mae={}_rmse={}.pth'
else:
    raise ValueError('Model not found')

model_files = os.listdir('model')
if param_config['model'] == 'CERNN':
    file = [value for value in model_files if f'{param_config["model"]}_K={param_config["K"]}' in value]
else:
    file = [value for value in model_files if f'{param_config["model"]}_τ={param_config["expectile_level"]}' in value]
if len(file):
    os.remove(os.path.join('model', file[0]))

if param_config['model'] == 'CERNN':
    expectile_levels = torch.tensor([k / (param_config['K'] + 1) for k in range(1, param_config['K'] + 1)],
                                    device=param_config['device'])

if param_config['model'] == 'CERNN':
    train_data = CaliforniaHousing(param_config['K'])
    test_data = CaliforniaHousing(param_config['K'], train=False)
else:
    train_data = CaliforniaHousing()
    test_data = CaliforniaHousing(train=False)
train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=param_config['batch_size'], shuffle=False)

best_bic = float('inf')
best_model = None
best_params = None
best_loss = None

for hidden_size, l2_lambda in product(param_config['hidden_size_range'], param_config['l2_lambda_range']):

    l2_lambda = np.round(l2_lambda, 4)

    if param_config['model'] == 'CERNN':
        model = CERNN(8, hidden_size, param_config['K']).to(param_config['device'])
        criterion = CERNN_LOSS(expectile_levels.cpu()).to(param_config['device'])
    elif param_config['model'] == 'ERNN':
        model = ERNN(8, hidden_size).to(param_config['device'])
        criterion = ExpectileLossWithPenalty(param_config['expectile_level']).to(param_config['device'])
    else:
        model = ANN(8, hidden_size).to(param_config['device'])
        criterion = nn.MSELoss().to(param_config['device'])
    optimizer = optim.Adam(model.parameters(), lr=param_config['learning_rate'])

    pbar = tqdm(range(param_config['epochs']))
    good_loss = float('inf')
    good_model = None
    model.train()
    for epoch in pbar:

        pbar.set_description(f'第{epoch + 1}轮训练')
        if param_config['model'] == 'CERNN':
            pbar.set_postfix_str(f'K={param_config["K"]} hidden_size={hidden_size} λ={l2_lambda}')
        else:
            pbar.set_postfix_str(f'τ={param_config["expectile_level"]} hidden_size={hidden_size} λ={l2_lambda}')

        for tensors, targets in train_data_loader:

            tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
            output = model(tensors)

            optimizer.zero_grad()
            if param_config['model'] in ['CERNN', 'ERNN']:
                loss = criterion(output, targets, model, l2_lambda / hidden_size / 8)
            else:
                loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            if loss.item() < good_loss:
                good_model = model
                good_loss = loss.item()

    model = good_model
    model.eval()
    mae_loss = []
    mse_loss = []
    bic_loss = []
    with torch.no_grad():
        for tensors, targets in train_data_loader:

            tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
            output = model(tensors)

            mae_loss.append(torch.abs(output - targets))
            mse_loss.append((output - targets) ** 2)

            if param_config['model'] in ['CERNN', 'ERNN']:
                loss = criterion(output, targets, model, 0)
            else:
                loss = criterion(output, targets)
            bic_loss.append(loss.item() * len(output))

    mae_loss = torch.mean(torch.cat(mae_loss, dim=0)).item()
    mse_loss = torch.mean(torch.cat(mse_loss, dim=0)).item()
    bic_loss = np.mean(bic_loss)

    df = 10 * hidden_size + 1
    bic = np.log(bic_loss) + np.log(12384) / 12384 * df / 2

    print(f'BIC: {bic:.4f}, RMSE: {np.sqrt(mse_loss):.4f}, MAE: {mae_loss:.4f}')
    time.sleep(0.1)

    if bic < best_bic:
        best_bic = bic
        best_model = model
        best_params = (hidden_size, l2_lambda)
        best_loss = (mae_loss, np.sqrt(mse_loss))

if param_config['model'] == 'CERNN':
    save_path = param_config['model_save_path'].format(param_config['model'],
                                                       param_config['K'], *best_params, *best_loss)
else:
    save_path = param_config['model_save_path'].format(param_config['model'], param_config['expectile_level'],
                                                       *best_params, *best_loss)
torch.save(best_model.state_dict(), save_path)
print(f'Best BIC: {best_bic:.4f} with hidden size {best_params[0]} and lambda {best_params[1]}')
