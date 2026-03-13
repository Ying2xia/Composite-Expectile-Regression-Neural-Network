import os
import time
from ANN import *
from tqdm import tqdm
from torch import optim
from generate_data import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


param_config = {
    'epochs': 1000,
    'seed': 20241113,
    'batch_size': 64,
    'learning_rate': 5e-4,
    'distribution': 'norm',
    'train_num': 800,
    'expectile_level': 0.8,
    'hidden_size': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
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

# 生成数据
train_data = UnivariateData(param_config['train_num'], param_config['expectile_level'],
                            param_config['distribution'])
train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)

writer = SummaryWriter('logs')

model = ANN(1, param_config['hidden_size']).to(param_config['device'])
criterion = nn.MSELoss().to(param_config['device'])
optimizer = optim.Adam(model.parameters(), lr=param_config['learning_rate'])

best_loss = float('inf')
best_model = None

pbar = tqdm(range(param_config['epochs']))
model.train()
for epoch in pbar:

    pbar.set_description(f'第{epoch + 1}轮训练')
    pbar.set_postfix_str(f'τ={param_config["expectile_level"]} hidden_size={param_config["hidden_size"]}')

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

        writer.add_scalar('train', loss.item(), epoch + 1)

writer.close()
