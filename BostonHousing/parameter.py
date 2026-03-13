import os
import time
from ANN import *
from ERNN import *
from CERNN import *
from tqdm import tqdm
from torch import optim
from bostonhousing import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


param_config = {
    'K': 9,
    'epochs': 1000,
    'model': 'CERNN',
    'batch_size': 4,
    'hidden_size': 6,
    'learning_rate': 6.5e-4,
    'expectile_level': 0.8,
    'l2_lambda': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if os.path.exists('logs'):
    for value in os.listdir('logs'):
        os.remove(os.path.join('logs', value))
    time.sleep(0.5)
    os.rmdir('logs')
    time.sleep(0.5)

if param_config['model'] == 'CERNN':
    expectile_levels = torch.tensor([k / (param_config['K'] + 1) for k in range(1, param_config['K'] + 1)],
                                    device=param_config['device'])

if param_config['model'] == 'CERNN':
    train_data = BostonHousing(param_config['K'])
elif param_config['model'] in ['ERNN', 'ANN']:
    train_data = BostonHousing()
else:
    raise ValueError('Model not found')
train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)

writer = SummaryWriter('logs')

if param_config['model'] == 'CERNN':
    model = CERNN(13, param_config['hidden_size'], param_config['K']).to(param_config['device'])
    criterion = CERNN_LOSS(expectile_levels.cpu()).to(param_config['device'])
elif param_config['model'] == 'ERNN':
    model = ERNN(13, param_config['hidden_size']).to(param_config['device'])
    criterion = ExpectileLossWithPenalty(param_config['expectile_level']).to(param_config['device'])
else:
    model = ANN(13, param_config['hidden_size']).to(param_config['device'])
    criterion = nn.MSELoss().to(param_config['device'])
optimizer = optim.Adam(model.parameters(), lr=param_config['learning_rate'])

best_loss = float('inf')
best_model = None

pbar = tqdm(range(param_config['epochs']))
model.train()
for epoch in pbar:

    pbar.set_description(f'第{epoch + 1}轮训练')
    if param_config['model'] == 'CERNN':
        pbar.set_postfix_str(f'K={param_config["K"]} hidden_size={param_config["hidden_size"]} λ={param_config["l2_lambda"]}')
    else:
        pbar.set_postfix_str(f'τ={param_config["expectile_level"]} hidden_size={param_config["hidden_size"]} λ={param_config["l2_lambda"]}')

    for tensors, targets in train_data_loader:

        tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])

        optimizer.zero_grad()
        output = model(tensors)
        if param_config['model'] in ['CERNN', 'ERNN']:
            loss = criterion(output, targets, model, param_config['l2_lambda'] / param_config['hidden_size'] / 13)
        else:
            loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_model = model
            best_loss = loss.item()

        writer.add_scalar('train', loss.item(), epoch + 1)

writer.close()
