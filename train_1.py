import os
import time
from CERNN import *
from tqdm import tqdm
from torch import optim
from generate_data import *
from itertools import product
from torch.utils.data import DataLoader

param_config = {
    'K': 9,
    'epochs': 1000,
    'batch_size': 8,
    'seed': 20241113,
    'train_num': 800,
    'early_stop': 20000,
    'learning_rate': 1e-3,
    'distribution': 'chi2',
    'hidden_size_range': list(range(1, 11)),
    'l2_lambda_range': np.arange(0, 0.0011, 0.0001),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_path': 'model/Univariate/K={}_distribution={}_hiddensize={}_λ={}_mae={}_rmse={}.pth'
}


def same_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


same_seed(param_config['seed'])
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('model/Univariate'):
    os.mkdir('model/Univariate')

model_files = os.listdir('model/Univariate')
file = [value for value in model_files if f'K={param_config["K"]}_distribution={param_config["distribution"]}' in value]
if len(file):
    os.remove(os.path.join('model/Univariate', file[0]))

expectile_levels = torch.tensor([k / (param_config['K'] + 1) for k in range(1, param_config['K'] + 1)],
                                device=param_config['device'])
train_data = UnivariateData(param_config['train_num'], expectile_levels.cpu(), param_config['distribution'])
train_data_loader = DataLoader(train_data, batch_size=param_config['batch_size'], shuffle=True)

best_bic = float('inf')
best_model = None
best_params = None
best_loss = None

for hidden_size, l2_lambda in product(param_config['hidden_size_range'], param_config['l2_lambda_range']):

    l2_lambda = np.round(l2_lambda, 4)

    model = CERNN(1, hidden_size, param_config['K']).to(param_config['device'])
    criterion = CERNN_LOSS(expectile_levels).to(param_config['device'])
    optimizer = optim.Adam(model.parameters(), lr=param_config['learning_rate'])

    pbar = tqdm(range(param_config['epochs']))
    good_loss = float('inf')
    good_model = None
    stop_count = 0
    model.train()
    for epoch in pbar:

        pbar.set_description(f'第{epoch + 1}轮训练')
        pbar.set_postfix_str(f'K={param_config["K"]} hidden_size={hidden_size} λ={l2_lambda}')

        for tensors, targets in train_data_loader:

            tensors, targets = tensors.to(param_config['device']), targets.to(param_config['device'])
            output = model(tensors)

            optimizer.zero_grad()
            loss = criterion(output, targets, model, l2_lambda / hidden_size)
            loss.backward()
            optimizer.step()

            if loss.item() < good_loss:
                good_model = model
                good_loss = loss.item()
                stop_count = 0

            else:
                stop_count += 1
                if stop_count >= param_config['early_stop']:
                    break

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

            loss = criterion(output, targets, model, 0)
            bic_loss.append(loss.item() * len(output))

        mae_loss = torch.mean(torch.cat(mae_loss, dim=0)).item()
        mse_loss = torch.mean(torch.cat(mse_loss, dim=0)).item()
        bic_loss = np.mean(bic_loss)

    df = 3 * hidden_size + 1
    bic = np.log(bic_loss) + np.log(param_config['train_num']) / param_config['train_num'] * df / 2

    print(f'BIC: {bic:.4f}, RMSE: {np.sqrt(mse_loss):.4f}, MAE: {mae_loss:.4f}')
    time.sleep(0.1)

    if bic < best_bic:
        best_bic = bic
        best_model = model
        best_params = (hidden_size, l2_lambda)
        best_loss = (mae_loss, np.sqrt(mse_loss))

torch.save(best_model.state_dict(),
           param_config['model_save_path'].format(param_config['K'], param_config['distribution'],
                                                  *best_params, *best_loss))
print(f'When K={param_config["K"]}, distribution is {param_config["distribution"]}, '
      f'Best BIC: {best_bic:.4f} with hidden size {best_params[0]} and lambda {best_params[1]}')
