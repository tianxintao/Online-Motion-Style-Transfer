import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os
import glob
from statistics import mean
from scipy import linalg

from model import RecurrentStylization, ContentClassification
from torch.utils.tensorboard import SummaryWriter
from dataloader import MotionDataset
from train import ArgParserTrain
from postprocess import save_bvh_from_network_output, remove_fs
from test import process_single_bvh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(edgeitems=5)

content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]

class ArgParserFID(ArgParserTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--train_autoencoder", default=False, action='store_true')
        self.add_argument("--aut_lr", default=1e-4, type=float)

        self.add_argument("--real_dir", default='./experiments/')
        self.add_argument("--fake_dir", default='./experiments/')
        
class DenoiseAutoencoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        dim = dim_dict["rotation"]
        self.encoder = nn.Sequential(
            nn.Conv1d(dim,128,4,stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,256,3,dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,512,2),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512,256,2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256,128,3,dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128,dim,4,stride=4)
            )

    def forward(self, rotation):
        batch_size, length, _ = rotation.shape
        latent_code = self.encoder(rotation.permute(0,2,1))
        rotations = self.decoder(latent_code).permute(0,2,1)
        rotation_norm = torch.norm(rotations.view(batch_size,length, -1, 4), dim=-1, keepdim=True)
        rotations = rotations.view(batch_size,length, -1, 4) / rotation_norm
        return rotations.view(batch_size, length, -1), latent_code

def main():
    args = ArgParserFID().parse_args()
    trainer = TrainerFID(args)
    if args.train_autoencoder:
        trainer.train_autoencoder()
    else:
        trainer.compute_score()

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def create_logger(output_path):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger_name = os.path.join(output_path, 'session.log')
    file_handler = logging.FileHandler(logger_name)
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    return logger

def get_style_name(style_vector):
    if style_vector.sum() == 0.0:
        style = "neutral"
    else:
        style = style_labels[(style_vector==1).nonzero(as_tuple=True)[0]]
    return style

class TrainerFID():
    def __init__(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H-%M", ts)
        exp_name = ts + '_' + 'seed_' + str(args.seed)
        exp_name = exp_name + '_' + args.tag if args.tag != '' else exp_name
        experiment_dir = os.path.join(args.work_dir, exp_name)

        make_dir(experiment_dir)
        self.video_dir = make_dir(os.path.join(experiment_dir, 'video'))
        self.model_dir = make_dir(os.path.join(experiment_dir, 'model'))
        self.buffer_dir = make_dir(os.path.join(experiment_dir, 'buffer'))
        self.args = args
        self.logger = create_logger(experiment_dir)

        with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        
        self.tb = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tb_logger'))

        self.dataset = MotionDataset(args)
        self.test_dataset = MotionDataset(args, subset_name="fid_test")
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=args.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=True)
        
        self.model = DenoiseAutoencoder(args, self.dataset.dim_dict).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.cla_lr)

        self.reset_loss_dict()

    def quaternion_difference(self, rot1, rot2):
        epsilon = 1e-7
        batch_size, duration, _ = rot1.shape
        rot1 = rot1.reshape(batch_size, duration, -1, 4)
        rot2 = rot2.reshape(batch_size, duration, -1, 4)
        angle = torch.acos(torch.clamp(torch.abs((rot1 * rot2).sum(dim=-1)), -1+epsilon, 1-epsilon))
        # angle = 2 * torch.acos(torch.abs(rot1 * rot2).sum(dim=-1), -1+epsilon, 1-epsilon)
        return (angle**2).sum(dim=-1).sum(dim=-1).mean()

    def reset_loss_dict(self):
        self.loss_dict = {
            "rotation_loss": []
        }

    def train_autoencoder(self):
        for epoch in range(self.args.n_epoch):
            for i, train_data in enumerate(self.dataloader):

                reconstructed_rotation = self.model(train_data["noisy_rotation"])
                rotation_loss = self.quaternion_difference(reconstructed_rotation, train_data["rotation"])
                self.loss_dict["rotation_loss"].append(rotation_loss.item())

                self.optimizer.zero_grad()
                rotation_loss.backward()
                self.optimizer.step()

                if (i+1) % self.args.log_freq == 0:
                    self.logger.info('Train: Epoch [{}/{}], Step [{}/{}]| r_loss: {:.3f}' 
                        .format(epoch+1, self.args.n_epoch, i+1, len(self.dataloader),mean(self.loss_dict["rotation_loss"]))
                        )
                    self.reset_loss_dict()
                    
            if (epoch+1) % self.args.test_freq == 0:
                self.model.eval()
                self.reset_loss_dict()
                j = np.random.randint(0, len(self.testloader), size=2)

                for i, test_data in enumerate(self.testloader):
                    reconstructed_motion = self.model(test_data["noisy_rotation"])
                    rotation_loss = self.quaternion_difference(reconstructed_motion, test_data["rotation"])
                    self.loss_dict["rotation_loss"].append(rotation_loss.item())
                    if i in j:
                        duration = reconstructed_motion.shape[1]
                        reconstructed_motion = reconstructed_motion.squeeze(0)
                        root_info = test_data["root"].squeeze(0).transpose(0, 1)
                        reconstructed_motion = torch.cat((reconstructed_motion,root_info), dim=-1).transpose(0, 1).detach().cpu()
                        
                        content = content_labels[(test_data["content"][0, 0, :]==1).nonzero(as_tuple=True)[0]]

                        input_style = test_data["input_style"][0, 0, :]
                        input_style = get_style_name(input_style)

                        save_bvh_from_network_output(
                            reconstructed_motion, 
                            os.path.join(self.video_dir, "{}_{}_{}_{}.bvh".format(epoch+1, i, input_style, content))
                        )

                        
                self.logger.info('Test: Epoch [{}/{}]| r_loss: {:.3f}' 
                        .format(epoch+1, self.args.n_epoch, mean(self.loss_dict["rotation_loss"]))
                        )
                self.reset_loss_dict()
                self.model.train()

            if (epoch+1) % self.args.save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model_{}.pt".format(epoch+1)))
    
    def get_latent_code(self, dir):
        result = []
        bvh_files = glob.glob(dir + '/*.bvh')
        for bvh in bvh_files:
            data = process_single_bvh(bvh, self.args, downsample=1, contain_label=False)
            _, latent_code = self.model(data["rotation"].unsqueeze(0))
            result.append(latent_code.squeeze(-1))
        return torch.stack(result).squeeze(1).detach().cpu().numpy()

    def compute_statistics(self, code):
        return np.mean(code, axis=0), np.cov(code, rowvar=False)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def compute_score(self):

        self.model.load_state_dict(torch.load(os.path.join(self.args.load_dir, 'model/model_1000.pt')))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        real_code = self.get_latent_code(self.args.real_dir)
        fake_code = self.get_latent_code(self.args.fake_dir)

        real_mean, real_cov = self.compute_statistics(real_code)
        fake_mean, fake_cov = self.compute_statistics(fake_code)

        print(self.calculate_frechet_distance(real_mean,real_cov,fake_mean,fake_cov))




if __name__ == "__main__":
    main()
        