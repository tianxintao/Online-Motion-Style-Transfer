import argparse
import json
import os
import torch
import time
import numpy as np
import os
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

from model import RecurrentStylization, ContentClassification
from utils.utils import make_dir, get_style_name, create_logger
from dataloader import MotionDataset
from postprocess import save_bvh_from_network_output, remove_fs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_labels = ["walk", "run", "jump", "punch", "kick"]
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]


class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup hyperparameters
        self.add_argument('--n_epoch', type=int, default=100)
        self.add_argument('--episode_length', type=int, default=24)
        self.add_argument("--seed", default=88, type=int)
        self.add_argument('--test_freq', type=int, default=50)
        self.add_argument('--log_freq', type=int, default=30)
        self.add_argument('--save_freq', type=int, default=50)
        self.add_argument('--style_num', type=int, default=7)
        self.add_argument('--content_num', type=int, default=5)
        self.add_argument("--data_path", default='./data/xia.npz')
        self.add_argument("--work_dir", default='./experiments/')
        self.add_argument("--load_dir", default=None, type=str)
        self.add_argument("--tag", default='', type=str)
        self.add_argument("--train_classifier", default=False, action='store_true')
        self.add_argument("--test_mode", default=False, action='store_true')

        # training hyperparameters
        self.add_argument("--batch_size", default=16, type=int)
        self.add_argument("--w_reg", default=10, type=int)
        self.add_argument("--dis_lr", default=5e-5, type=float)
        self.add_argument("--gen_lr", default=1e-4, type=float)
        self.add_argument("--cla_lr", default=1e-4, type=float)
        self.add_argument("--perceptual_loss", default=False, action='store_true')

        # model hyperparameters
        self.add_argument("--no_pos", default=False, action='store_true')
        self.add_argument("--no_vel", default=False, action='store_true')
        self.add_argument("--encoder_layer_num", type=int, default=2)
        self.add_argument("--decoder_layer_num", type=int, default=4)
        self.add_argument("--discriminator_layer_num", type=int, default=4)
        self.add_argument("--classifier_layer_num", type=int, default=5)
        self.add_argument("--latent_dim", type=int, default=32)
        self.add_argument("--neutral_layer_num", type=int, default=4)
        self.add_argument("--style_layer_num", type=int, default=6)
        self.add_argument("--feature_dim", type=int, default=16)


def main():
    args = ArgParserTrain().parse_args()
    trainer = Trainer(args)
    if args.train_classifier:
        trainer.train_classifier()
    elif args.test_mode:
        trainer.test()
    else:
        trainer.train()


class Trainer():
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
        self.test_dataset = MotionDataset(args, subset_name="test")
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=args.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=True)

        if args.train_classifier:
            self.model = ContentClassification(args, self.dataset.dim_dict).to(device)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.cla_lr)
        else:
            self.model = RecurrentStylization(args, self.dataset.dim_dict).to(device)
            if self.args.perceptual_loss:
                self.classification = ContentClassification(args, self.dataset.dim_dict).to(device)
                self.classification.load_state_dict(torch.load('./data/classifier.pt'))
                self.classification.eval()
                for param in self.classification.parameters():
                    param.requires_grad = False
            self.criterion = torch.nn.MSELoss()
            self.gen_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=args.gen_lr)
            self.dis_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=args.dis_lr)

        self.reset_loss_dict()

    def train_discriminator(self, train_data):

        real_score, grad = self.model.forward_dis(train_data["rotation"], train_data["position"],
                                                  train_data["velocity"], train_data["input_style"],
                                                  train_data["content"], compute_grad=True)
        grad_loss = self.args.w_reg * grad
        _, fake_score = self.model(train_data["rotation"], train_data["position"], train_data["velocity"],
                                   train_data["content"], train_data["contact"], train_data["root"],
                                   train_data["input_style"], train_data["transferred_style"])

        real_loss = self.criterion(real_score, torch.ones_like(real_score))
        fake_loss = self.criterion(fake_score, -torch.ones_like(fake_score))
        self.loss_dict["gradient_loss"].append(grad_loss.item())

        return real_loss + fake_loss + grad_loss

    def train_gen(self, train_data):

        # reconstruction
        reconstructed_motion = self.model.forward_gen(train_data["rotation"], train_data["position"],
                                                      train_data["velocity"],
                                                      train_data["content"], train_data["contact"],
                                                      train_data["input_style"], train_data["input_style"])

        # fake motion
        fake_motion, fake_score = self.model(train_data["rotation"], train_data["position"], train_data["velocity"],
                                             train_data["content"], train_data["contact"], train_data["root"],
                                             train_data["input_style"], train_data["transferred_style"])

        return self.compute_generator_loss(reconstructed_motion, fake_motion, train_data, fake_score)

    def compute_generator_loss(self, reconstruction, fake_motion, ground_truth, fake_score):

        rotation_loss = self.quaternion_difference(reconstruction["rotation"], ground_truth["rotation"]) * 0.05
        # rotation_loss = self.criterion(reconstruction["rotation"], ground_truth["rotation"]) * 64.0
        position_loss = self.criterion(reconstruction["position"], ground_truth["position"])
        velocity_loss = self.criterion(reconstruction["velocity"], ground_truth["velocity"])

        perceptual_loss = torch.tensor(0.0, device=device)
        if fake_motion != None:
            adversarial_loss = self.criterion(fake_score, torch.zeros_like(fake_score))
            if self.args.perceptual_loss:
                _, ground_truth_features = self.classification(ground_truth["rotation"], ground_truth["position"],
                                                               ground_truth["velocity"])
                _, fake_motion_features = self.classification(fake_motion["rotation"], fake_motion["position"],
                                                              fake_motion["velocity"])
                for ground_truth_key, fake_motion_key in zip(ground_truth_features.keys(), fake_motion_features.keys()):
                    perceptual_loss = perceptual_loss + self.criterion(ground_truth_features[ground_truth_key],
                                                                       fake_motion_features[fake_motion_key])
        else:
            adversarial_loss = torch.tensor(0)

        self.loss_dict["rotation_loss"].append(rotation_loss.item())
        self.loss_dict["position_loss"].append(position_loss.item())
        self.loss_dict["velocity_loss"].append(velocity_loss.item())
        self.loss_dict["perceptual_loss"].append(perceptual_loss.item())
        self.loss_dict["adversarial_loss"].append(adversarial_loss.item())
        generator_loss = 1.0 * rotation_loss + 0.5 * position_loss + velocity_loss + 1.0 * adversarial_loss + 0.1 * perceptual_loss
        self.loss_dict["generator_loss"].append(generator_loss.item())
        return generator_loss

    def quaternion_difference(self, rot1, rot2):
        epsilon = 1e-7
        batch_size, duration, _ = rot1.shape
        rot1 = rot1.reshape(batch_size, duration, -1, 4)
        rot2 = rot2.reshape(batch_size, duration, -1, 4)
        angle = torch.acos(torch.clamp(torch.abs((rot1 * rot2).sum(dim=-1)), -1 + epsilon, 1 - epsilon))
        return (angle ** 2).sum(dim=-1).sum(dim=-1).mean()

    def reset_loss_dict(self):
        self.loss_dict = {
            "generator_loss": [],
            "rotation_loss": [],
            "position_loss": [],
            "velocity_loss": [],
            "adversarial_loss": [],
            "perceptual_loss": [],
            "discriminator_loss": [],
            "gradient_loss": [],
            "crossentropy_loss": []
        }

    def train(self):
        for epoch in range(self.args.n_epoch):
            for i, train_data in enumerate(self.dataloader):

                discriminator_loss = self.train_discriminator(train_data)
                self.dis_optimizer.zero_grad()
                discriminator_loss.backward()
                self.dis_optimizer.step()
                self.loss_dict["discriminator_loss"].append(discriminator_loss.item())

                generator_loss = self.train_gen(train_data)
                self.gen_optimizer.zero_grad()
                generator_loss.backward()
                self.gen_optimizer.step()

                if (i + 1) % self.args.log_freq == 0:
                    self.logger.info(
                        'Train: Epoch [{}/{}], Step [{}/{}]| g_loss: {:.3f}| d_loss: {:.3f}| gp_loss: {:.3f}| r_loss: {:.3f}| p_loss: {:.3f}| v_loss: {:.3f}| per_loss: {:.3f} | a_loss: {:.3f}'
                            .format(epoch + 1, self.args.n_epoch, i + 1, len(self.dataloader),
                                    mean(self.loss_dict["generator_loss"]),
                                    mean(self.loss_dict["discriminator_loss"]),
                                    mean(self.loss_dict["gradient_loss"]),
                                    mean(self.loss_dict["rotation_loss"]),
                                    mean(self.loss_dict["position_loss"]),
                                    mean(self.loss_dict["velocity_loss"]),
                                    mean(self.loss_dict["perceptual_loss"]),
                                    mean(self.loss_dict["adversarial_loss"])
                                    ))
                    self.reset_loss_dict()

            if (epoch + 1) % self.args.test_freq == 0:
                self.model.eval()
                j = np.random.randint(0, len(self.testloader), size=5)

                for i, test_data in enumerate(self.testloader):
                    reconstructed_motion = self.model.forward_gen(test_data["rotation"], test_data["position"],
                                                                  test_data["velocity"],
                                                                  test_data["content"], test_data["contact"],
                                                                  test_data["input_style"], test_data["input_style"])
                    _ = self.compute_generator_loss(reconstructed_motion, None, test_data, None)
                    if i in j:
                        reconstructed_motion = reconstructed_motion["rotation"].squeeze(0)
                        root_info = test_data["root"].squeeze(0).transpose(0, 1)
                        reconstructed_motion = torch.cat((reconstructed_motion, root_info), dim=-1).transpose(0,
                                                                                                              1).detach().cpu()

                        content = content_labels[(test_data["content"][0, 0, :] == 1).nonzero(as_tuple=True)[0]]

                        input_style = test_data["input_style"][0, 0, :]
                        input_style = get_style_name(input_style)

                        save_bvh_from_network_output(
                            reconstructed_motion,
                            os.path.join(self.video_dir, "{}_{}_{}_{}.bvh".format(epoch + 1, i, input_style, content))
                        )

                        transferred_motion = self.model.forward_gen(test_data["rotation"], test_data["position"],
                                                                    test_data["velocity"],
                                                                    test_data["content"], test_data["contact"],
                                                                    test_data["input_style"],
                                                                    test_data["transferred_style"])
                        transferred_motion = transferred_motion["rotation"].squeeze(0)
                        transferred_motion = torch.cat((transferred_motion, root_info), dim=-1).transpose(0,
                                                                                                          1).detach().cpu()
                        transferred_style = get_style_name(test_data["transferred_style"][0, 0, :])

                        save_bvh_from_network_output(
                            transferred_motion,
                            os.path.join(self.video_dir,
                                         "{}_{}_{}_to_{}_{}.bvh".format(epoch + 1, i, input_style, transferred_style,
                                                                        content))
                        )

                self.logger.info('Test: Epoch [{}/{}]| g_loss: {:.3f}| r_loss: {:.3f}| p_loss: {:.3f}| v_loss: {:.3f}'
                                 .format(epoch + 1, self.args.n_epoch,
                                         mean(self.loss_dict["generator_loss"]),
                                         mean(self.loss_dict["rotation_loss"]),
                                         mean(self.loss_dict["position_loss"]),
                                         mean(self.loss_dict["velocity_loss"]),
                                         ))
                self.reset_loss_dict()
                self.model.train()

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model_{}.pt".format(epoch + 1)))

    def train_classifier(self):
        for epoch in range(self.args.n_epoch):
            for i, train_data in enumerate(self.dataloader):

                output_score, _ = self.model(train_data["rotation"], train_data["position"], train_data["velocity"])
                loss = self.criterion(output_score, train_data["content_index"].squeeze(1))
                self.loss_dict["crossentropy_loss"].append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % self.args.log_freq == 0:
                    self.logger.info('Train: Epoch [{}/{}], Step [{}/{}]| loss: {:.4f}'
                                     .format(epoch + 1, self.args.n_epoch, i + 1, len(self.dataloader),
                                             mean(self.loss_dict["crossentropy_loss"])))
                    self.reset_loss_dict()

            if (epoch + 1) % self.args.test_freq == 0:
                self.model.eval()
                total = 0
                correct = 0
                for i, test_data in enumerate(self.testloader):
                    output_score, _ = self.model(test_data["rotation"], test_data["position"], test_data["velocity"])
                    _, predicted = torch.max(output_score, 1)
                    total += test_data["rotation"].size(0)
                    correct += (predicted == test_data["content_index"]).sum().item()

                self.logger.info('Test: Epoch [{}/{}]| accuracy: {:.4f}%'
                                 .format(epoch + 1, self.args.n_epoch, correct / total))
                self.model.train()

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model_{}.pt".format(epoch + 1)))

    def test(self):

        self.model.load_state_dict(torch.load(os.path.join(self.args.load_dir, 'model/model_2000.pt')))
        self.model.eval()
        # for test_data in self.test_dataset:
        # for selected_index in [1002,1003,1004,1005,1006,1007,1008,1080]:
        for selected_index in [24]:
            test_data = self.test_dataset[selected_index]
            for (key, val) in test_data.items():
                if key != "content_index":
                    test_data[key] = val[0].unsqueeze(0).unsqueeze(0)

            content = content_labels[(test_data["content"][0, 0, :] == 1).nonzero(as_tuple=True)[0]]
            input_style = test_data["input_style"][0, 0, :]
            if input_style.sum() == 0.0:
                input_style = "neutral"
            else:
                input_style = style_labels[(input_style == 1).nonzero(as_tuple=True)[0]]
            for ind in range(7):
                output_style_index = ind
                output_style = style_labels[output_style_index]
                test_data["transferred_style"] = torch.zeros_like(test_data["transferred_style"])
                test_data["transferred_style"][..., output_style_index] = 1
                # test_data["transferred_style"] = test_data["input_style"]

                start_time = time.time()
                for _ in range(100):
                    transferred_motion = self.model.forward_gen(test_data["rotation"], test_data["position"],
                                                                test_data["velocity"],
                                                                test_data["content"], test_data["contact"],
                                                                test_data["input_style"], test_data["transferred_style"],
                                                                test_time=True)
                end_time = time.time()
                print(end_time-start_time)

                transferred_motion = transferred_motion["rotation"].squeeze(0)
                root_info = test_data["root"].squeeze(0).transpose(0, 1)
                foot_contact = test_data["contact"].cpu().squeeze(0).transpose(0, 1).numpy()
                transferred_motion = torch.cat((transferred_motion, root_info), dim=-1).transpose(0, 1).detach().cpu()
                save_bvh_from_network_output(
                    transferred_motion,
                    os.path.join(self.args.load_dir,
                                 "test/{}_to_{}_{}_{}.bvh".format(input_style, output_style, content, selected_index))
                )
                remove_fs(
                    transferred_motion,
                    foot_contact,
                    output_path=os.path.join(self.args.load_dir,
                                             "submission/{}_to_{}_{}_{}.bvh".format(input_style, output_style, content,
                                                                                    selected_index))
                )

                # original_motion = torch.cat((test_data["rotation"][0], root_info), dim=-1).transpose(0, 1).detach().cpu()
                # save_bvh_from_network_output(
                #     original_motion, 
                #     os.path.join(self.args.load_dir, "submission/{}_to_{}_{}_{}.bvh".format(input_style, "original", content, selected_index))
                # ) 


if __name__ == "__main__":
    main()
