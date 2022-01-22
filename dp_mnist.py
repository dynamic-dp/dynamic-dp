import math
import numpy as np
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.GaussianCalibrator import calibrateAnalyticGaussianMechanism
import math
from utils.poisson_sampler import poisson_sampler
from utils.mu_search import mu0_search,cal_step_decay_rate
from scipy.stats import norm
from scipy import optimize

"""
Runs MNIST training with differential privacy.
"""



# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps,mu):
    return norm.cdf(-eps/mu+mu/2)-np.exp(eps)*norm.cdf(-eps/mu-mu/2)
# inverse Dual
def eps_from_mu(mu,delta):
    def f(x):
        return delta_eps_mu(x,mu)-delta    
    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x
    def name(self):
        return "SampleConvNet"

def train(args, step, model, device, train_loader, optimizer, 
train_dataset = False, dp=False ,sens_decay = False, 
mu_allocation = False, privacy_engine =None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0 
    total = 0
    if dp == False:
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            pred = output.argmax(
                                dim=1, keepdim=True
                            ) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    else:
        if sens_decay:
            clip = args.max_per_sample_grad_norm * (args.decay_rate_sens)**step
            privacy_engine.set_clip(clip)
        if mu_allocation:
            unit_sigma = 1/(args.mu_0/(args.decay_rate_mu**(step)))
            privacy_engine.set_unit_sigma(unit_sigma)
        for i in tqdm(range(int(1/args.sampling_rate))):
            data,target = poisson_sampler(train_dataset,args.sampling_rate)
            data, target = data.to(device), target.to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            pred = output.argmax(
                            dim=1, keepdim=True
                        ) 
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.shape[0]
        acc = 100.0*correct/ total
    return step

def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def main(dp = False, epsilon = None, sens_decay = False, mu_allocation = False,n_runs=1,
    decay_rate_sens = None, decay_rate_mu = None):
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    # Training settings
    lr = 0.15
    epochs = 20
    num_data = 60000
    batch_size = 256 # this is the expectated batch size since we use poisson
    sampling_rate = batch_size/num_data
    iteration = int(epochs/sampling_rate)
    if epsilon:
        clipping_value = 1.5
        delta = 1e-5
        mu = 1/calibrateAnalyticGaussianMechanism(epsilon = epsilon, delta  = delta, GS = 1, tol = 1.e-12)
        mu_t = math.sqrt(math.log(mu**2/(sampling_rate**2*iteration)+1))
        sigma = 1/mu_t
        if mu_allocation:
            decay_rate_mu = cal_step_decay_rate(decay_rate_mu,iteration)
        if sens_decay:
            decay_rate_sens = cal_step_decay_rate(decay_rate_sens,iteration)
        parser.add_argument(
        "--mu_t",
        type=float,
        default=mu_t,
        )
        parser.add_argument(
            "--sigma",
            type=float,
            #default=1.0,
            default=sigma,
            metavar="S",
            help="Noise multiplier (default 1.0)",
            )
        parser.add_argument(
            "-c",
            "--max-per-sample-grad_norm",
            type=float,
            default=clipping_value,
            metavar="C",
            help="Clip per-sample gradients to this norm (default 1.0)",
        )
        parser.add_argument(
            "--delta",
            type=float,
            #default=1e-5,
            default=delta,
            metavar="D",
            help="Target delta (default: 1e-5)",
        )
        if decay_rate_mu:
            mu_0 = mu0_search(mu,iteration,decay_rate_mu,sampling_rate,mu_t=mu_t)
            parser.add_argument(
                "--mu_0",
                type=float,
                default=mu_0,
            )

    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=sampling_rate,
    )
    parser.add_argument(
        "--decay_rate_sens",
        type=float,
        default=decay_rate_sens,
    )
    parser.add_argument(
        "--decay_rate_mu",
        type=float,
        default=decay_rate_mu,
        help='decay rate of 1/mu'
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        #default=64,
        default=batch_size,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=epsilon,
        metavar="EP",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        #default=2,
        default=epochs,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=n_runs,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        #default=0.1,
        default=lr,
        metavar="LR",
        help="learning rate (default: .1)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        default=dp,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    kwargs = {"num_workers": 1, "pin_memory": True}

    train_dataset = datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    step = 0
    model = SampleConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    privacy_engine = None
    if dp:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=sampling_rate,
            max_grad_norm=args.max_per_sample_grad_norm,
            noise_multiplier= sigma,
        )
        privacy_engine.attach(optimizer)
    for epoch in range(1, args.epochs + 1):
        step = train(args, step,model, device, train_loader, optimizer, train_dataset=train_dataset, 
        dp= dp, sens_decay = sens_decay, mu_allocation = mu_allocation,privacy_engine=privacy_engine
        )
        test(args,model,device,test_loader=test_loader)
    return(test(args,model,device,test_loader=test_loader))

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    #acc_no_dp = main(dp=False)
    for ep in [0.4]:
        # dr_sens = np.linspace(0.7,0.8,2)
        # dr_mus = np.linspace(0.2,0.6,5)
        dr_sens = [0.5]
        dr_mus = [0.7]
        acc_dp = main(dp = True,epsilon = ep)
        for dr_sen in dr_sens:
            for dr_mu in dr_mus:
                acc_dynamic_dp = main(dp = True , epsilon = ep,sens_decay=True, mu_allocation=True,decay_rate_sens=dr_sen,decay_rate_mu=dr_mu)
    # print('Acc without dp: ',acc_no_dp,'Acc with dp: ',acc_dp,'Acc with dynamic dp: ',acc_dynamic_dp)
    print('Acc with uniform dp: ',acc_dp,'Acc with dynamic dp: ',acc_dynamic_dp)