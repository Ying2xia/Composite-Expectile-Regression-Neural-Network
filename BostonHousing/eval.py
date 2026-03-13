import os
import re
import time
from ANN import *
from ERNN import *
from CERNN import *
from bostonhousing import *
from torch.utils.data import DataLoader

param_config = {
    'K': 5,
    'model': 'ERNN',
    'batch_size': 4,
    'expectile_level': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

best_model = [value for value in os.listdir('model') if value.startswith(param_config['model'])]
if param_config['model'] == 'CERNN':
    best_model = [value for value in best_model if f'K={param_config["K"]}' in value][0]
elif param_config['model'] in ['ERNN', 'ANN']:
    best_model = [value for value in best_model if f'τ={param_config["expectile_level"]}' in value][0]
else:
    raise ValueError('Model not found')
print(best_model)
time.sleep(0.1)

pattern_1 = re.compile(r'_hiddensize=(\d+)_')
pattern_2 = re.compile(r'_λ=(0.\d+)_')
hidden_size = eval(pattern_1.findall(best_model)[0])
l2_lambda = 0 if param_config['model'] == 'ANN' else eval(pattern_2.findall(best_model)[0])

if param_config['model'] == 'CERNN':
    expectile_levels = torch.tensor([k / (param_config['K'] + 1) for k in range(1, param_config['K'] + 1)],
                                    device=param_config['device'])

if param_config['model'] == 'CERNN':
    train_data = BostonHousing(param_config['K'])
    test_data = BostonHousing(param_config['K'], train=False)
else:
    train_data = BostonHousing()
    test_data = BostonHousing(train=False)
train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=param_config['batch_size'], shuffle=False)

if param_config['model'] == 'CERNN':
    model = CERNN(13, hidden_size, param_config['K']).to(param_config['device'])
elif param_config['model'] == 'ERNN':
    model = ERNN(13, hidden_size).to(param_config['device'])
else:
    model = ANN(13, hidden_size).to(param_config['device'])
model.load_state_dict(torch.load(os.path.join('model', best_model), weights_only=True))

model.eval()
mae_loss = []
mse_loss = []
with torch.no_grad():
    for tensors, targets in train_data_loader:

        tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
        output = model(tensors)

        mae_loss.append(torch.abs(output - targets))
        mse_loss.append((output - targets) ** 2)

mae_loss = torch.mean(torch.cat(mae_loss, dim=0)).item()
mse_loss = torch.mean(torch.cat(mse_loss, dim=0)).item()
print(f'Train MAE: {mae_loss}, Train RMSE: {np.sqrt(mse_loss)}')

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
print(f'Test MAE: {mae_loss}, Test RMSE: {np.sqrt(mse_loss)}')
