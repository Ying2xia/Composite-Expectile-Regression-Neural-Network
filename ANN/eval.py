import os
import re
import time
from ANN import *
from tqdm import trange
from torch import optim
from generate_data import *
from torch.utils.data import DataLoader

param_config = {
    'epochs': 1000,
    'input_dim': 4,
    'test_num': 200,
    'train_num': 800,
    'batch_size': 96,
    'early_stop': 1800,
    'expectile_level': 0.2,
    'distribution': 'chi2',
    'learning_rate': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

generator = {
    1: UnivariateData,
    3: TrivariateData,
    4: QuadrivariateData
}
file = {
    1: 'model/Univariate',
    3: 'model/Trivariate',
    4: 'model/Quadrivariate'
}

best_model = [value for value in os.listdir(file[param_config['input_dim']]) if value.endswith('.pth')]
best_model = [value for value in best_model
              if f'τ=0.8_distribution={param_config["distribution"]}' in value][0]
print(best_model)
time.sleep(0.1)

pattern = re.compile(r'_hiddensize=(\d+)_')
hidden_size = pattern.findall(best_model)[0]

train_loss = []
test_loss = []
start_time = time.time()
for _ in trange(999):

    model = ANN(param_config['input_dim'], eval(hidden_size)).to(param_config['device'])
    criterion = nn.MSELoss().to(param_config['device'])
    optimizer = optim.Adam(model.parameters(), lr=param_config['learning_rate'])

    train_data = generator[param_config['input_dim']](param_config['train_num'], param_config['expectile_level'],
                                                      distribution=param_config['distribution'])
    test_data = generator[param_config['input_dim']](param_config['test_num'], param_config['expectile_level'],
                                                     distribution=param_config['distribution'])
    train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=param_config['batch_size'], shuffle=False)

    stop_count = 0
    best_loss = float('inf')
    best_model = None
    for epoch in range(param_config['epochs']):
        for tensors, targets in train_data_loader:

            tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
            optimizer.zero_grad()
            output = model(tensors)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_model = model
                best_loss = loss.item()
                stop_count = 0

            else:
                stop_count += 1
                if stop_count >= param_config['early_stop']:
                    break

    model = best_model
    mae_loss = []
    mse_loss = []
    model.eval()
    with torch.no_grad():
        for tensors, targets in train_data_loader:

            tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
            output = model(tensors)

            mae_loss.append(torch.abs(output - targets))
            mse_loss.append((output - targets) ** 2)

    mae_loss = torch.mean(torch.cat(mae_loss, dim=0)).item()
    mse_loss = torch.mean(torch.cat(mse_loss, dim=0)).item()
    train_loss.append([np.mean(mae_loss), np.sqrt(np.mean(mse_loss))])

    mae_loss = []
    mse_loss = []
    with torch.no_grad():
        for tensors, targets in test_data_loader:

            tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
            output = model(tensors)

            mae_loss.append(torch.abs(output - targets))
            mse_loss.append((output - targets) ** 2)

    mae_loss = torch.mean(torch.cat(mae_loss, dim=0)).item()
    mse_loss = torch.mean(torch.cat(mse_loss, dim=0)).item()
    test_loss.append([np.mean(mae_loss), np.sqrt(np.mean(mse_loss))])

print(f'Time cost: {time.time() - start_time}')

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
np.save(f'data/{param_config["input_dim"]}{param_config["expectile_level"]}train_{param_config["distribution"]}.npy', train_loss)
np.save(f'data/{param_config["input_dim"]}{param_config["expectile_level"]}test_{param_config["distribution"]}.npy', test_loss)
print(f'In sample, MAE: {np.mean(train_loss[:, 0]):.6f}({np.std(train_loss[:, 0]):.6f}), '
      f'RMSE: {np.mean(train_loss[:, 1]):.6f}({np.std(train_loss[:, 1]):.6f})')
print(f'Out of sample, MAE: {np.mean(test_loss[:, 0]):.6f}({np.std(test_loss[:, 0]):.6f}), '
      f'RMSE: {np.mean(test_loss[:, 1]):.6f}({np.std(test_loss[:, 1]):.6f})')
