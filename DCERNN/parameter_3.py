import os
import time
from CERNN import *
from tqdm import tqdm
from torch import optim
from generate_data import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# change K, distribution and data_generator every time before start
param_config = {
    'K': 9,
    'epochs': 1000,
    'seed': 20241113,
    'train_num': 8000,
    'batch_size': 640,
    'learning_rate': 2e-3,
    'distribution': 'norm',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'hidden_size': 9,
    'l2_lambda': 0.0005
}


def same_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


same_seed(param_config['seed'])
if os.path.exists('logs'):
    for value in os.listdir('logs'):
        os.remove(os.path.join('logs', value))
    time.sleep(0.5)
    os.rmdir('logs')
    time.sleep(0.5)

expectile_levels = torch.tensor([k / (param_config['K'] + 1) for k in range(1, param_config['K'] + 1)],
                                device=param_config['device'])
train_data = TrivariateData(param_config['train_num'], expectile_levels.cpu(), param_config['distribution'])
train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)

best_loss = float('inf')
best_model = None
writer = SummaryWriter('logs')

model = CERNN(3, param_config['hidden_size'], param_config['K']).to(param_config['device'])
criterion = CERNN_LOSS(expectile_levels).to(param_config['device'])
optimizer = optim.Adam(model.parameters(), lr=param_config['learning_rate'])

pbar = tqdm(range(param_config['epochs']))
model.train()
for epoch in pbar:

    pbar.set_description(f'第{epoch + 1}轮训练')
    pbar.set_postfix_str(f'K={param_config["K"]} hidden_size={param_config["hidden_size"]} λ={param_config["l2_lambda"]}')

    for tensors, targets in train_data_loader:

        tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
        output = model(tensors)

        optimizer.zero_grad()
        loss = criterion(output, targets, model, param_config['l2_lambda'] / param_config['hidden_size'] / 3)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train', loss.item(), epoch)

writer.close()
