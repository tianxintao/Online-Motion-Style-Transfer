import torch
import torch.nn as nn
from torch import autograd
from kinematics import ForwardKinematics


def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))


class Encoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        input_size = dim_dict["rotation"] + dim_dict["position"] + dim_dict["velocity"] + \
                     dim_dict["style"] + dim_dict["content"] + dim_dict["contact"]
        for _ in range(args.encoder_layer_num):
            layers.append(nn.Linear(input_size, input_size // 4 * 2))
            layers.append(nn.ReLU())
            input_size = input_size // 4 * 2

        layers.append(nn.Linear(input_size, args.latent_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, rotation, position, velocity, content, contact, input_style):
        return self.layers(torch.cat((rotation, position, velocity, contact, content, input_style), dim=-1))


class ResidualAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.neutral_branch = nn.LSTM(input_size=args.latent_dim, hidden_size=args.latent_dim,
                                      num_layers=args.neutral_layer_num, batch_first=True)
        self.neutral_init_hidden_state = nn.Parameter(
            torch.zeros(args.neutral_layer_num, args.content_num, args.latent_dim))
        self.neutral_init_cell_state = nn.Parameter(
            torch.zeros(args.neutral_layer_num, args.content_num, args.latent_dim))

        ra_style = []
        for _ in range(args.style_num):
            ra_style.append(
                nn.LSTM(input_size=args.latent_dim, hidden_size=args.latent_dim,
                        num_layers=args.style_layer_num, batch_first=True)
            )
        self.ra_init_hidden_state = nn.Parameter(torch.zeros(args.style_layer_num, args.style_num, args.latent_dim))
        self.ra_init_cell_state = nn.Parameter(torch.zeros(args.style_layer_num, args.style_num, args.latent_dim))
        self.ra_branch = nn.ModuleList(ra_style)

    def forward(self, latent_code, transferred_style, content, test_time=False):
        batch_size, T, _ = latent_code.shape
        content_index = torch.argmax(content.reshape(batch_size, T, -1).mean(dim=1), dim=-1)

        h0_neutral = self.neutral_init_hidden_state[:, content_index, :]
        c0_neutral = self.neutral_init_cell_state[:, content_index, :]
        neutral_output, (hn, cn) = self.neutral_branch(latent_code, (h0_neutral, c0_neutral))

        ra_result = []
        if test_time:
            transferred_style = transferred_style[:, 0, :]
            if transferred_style.flatten().sum() > 0:
                i = torch.argmax(transferred_style.flatten())
                branch = self.ra_branch[i]
                h0_ra = self.ra_init_hidden_state[:, i, :].unsqueeze(1).repeat(1, batch_size, 1)
                c0_ra = self.ra_init_cell_state[:, i, :].unsqueeze(1).repeat(1, batch_size, 1)
                ra_result = branch(latent_code, (h0_ra, c0_ra))[0]
                return neutral_output + ra_result
            else:
                return neutral_output
        else:
            for i, branch in enumerate(self.ra_branch):
                h0_ra = self.ra_init_hidden_state[:, i, :].unsqueeze(1).repeat(1, batch_size, 1)
                c0_ra = self.ra_init_cell_state[:, i, :].unsqueeze(1).repeat(1, batch_size, 1)
                ra_result.append(branch(latent_code, (h0_ra, c0_ra))[0])
            ra_value = torch.stack(ra_result) * transferred_style.permute(2, 0, 1).unsqueeze(-1)

            return neutral_output + ra_value.sum(dim=0)


class Decoder(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        current_dim = args.latent_dim + dim_dict["style"]
        self.fk = ForwardKinematics()
        self.args = args
        layers = []
        for _ in range(args.decoder_layer_num - 1):
            layers.append(nn.Linear(current_dim, int((current_dim * 1.5 // 2) * 2)))
            layers.append(nn.ReLU())
            current_dim = int((current_dim * 1.5 // 2) * 2)

        self.features = nn.Sequential(*layers)
        self.rotation_layer = nn.Linear(current_dim, dim_dict["rotation"])
        if not args.no_pos: self.position_layer = nn.Linear(current_dim, dim_dict["position"])
        if not args.no_vel: self.velocity_layer = nn.Linear(current_dim, dim_dict["velocity"])

    def forward(self, latent_code, style):
        style = torch.zeros_like(style)
        features = self.features(torch.cat((latent_code, style), dim=-1))

        output_dict = {}
        output_dict["rotation"] = self.rotation_layer(features)
        batch_size, length, rotation_dim = output_dict["rotation"].shape
        rotation_norm = torch.norm(output_dict["rotation"].view(batch_size, length, -1, 4), dim=-1, keepdim=True)
        output_dict["rotation"] = output_dict["rotation"].view(batch_size, length, -1, 4) / rotation_norm
        output_dict["rotation"] = output_dict["rotation"].view(batch_size, length, -1)

        if not self.args.no_pos:
            output_dict["position"] = self.position_layer(features)
        else:
            output_dict["position"] = self.fk.forwardX(output_dict["rotation"].permute(0, 2, 1)).permute(0, 2, 1)
        if not self.args.no_vel:
            output_dict["velocity"] = self.velocity_layer(features)
        else:
            velocity = output_dict["position"][:, 1:, :] - output_dict["position"][:, :-1, :]
            velocity_last = (2 * velocity[:, -1, :] - velocity[:, -2, :]).unsqueeze(1)
            output_dict["velocity"] = torch.cat((velocity, velocity_last), dim=1)

        return output_dict


class Generator(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args, dim_dict)
        self.ra = ResidualAdapter(args)
        self.decoder = Decoder(args, dim_dict)

    def forward(self, rotation, position, velocity, content, contact, input_style, transferred_style, test_time=False):
        batch_size, length, _ = rotation.shape

        rotation = rotation.reshape(-1, rotation.shape[-1])
        position = position.reshape(-1, position.shape[-1])
        velocity = velocity.reshape(-1, velocity.shape[-1])
        content = content.reshape(-1, content.shape[-1])
        contact = contact.reshape(-1, contact.shape[-1])
        input_style = input_style.reshape(-1, input_style.shape[-1])
        encoded_data = self.encoder(rotation, position, velocity, content, contact, input_style)
        latent_code = self.ra(encoded_data.view(batch_size, length, -1), transferred_style, content,
                              test_time=test_time)
        output = self.decoder(latent_code, transferred_style)

        return output


class Discriminator(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        current_size = dim_dict["position"] + dim_dict["velocity"]
        dummy_data = torch.zeros(1, current_size, args.episode_length)
        for _ in range(args.discriminator_layer_num):
            layers.append(conv_layer(3, current_size, (current_size // 3) * 2))
            layers.append(nn.LeakyReLU())
            current_size = (current_size // 3) * 2
        self.features = nn.Sequential(*layers)

        self.last_layer = conv_layer(3, current_size, args.feature_dim)

        input_size = dim_dict["style"] + dim_dict["content"]

        self.attention_features = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )
        self.temporal_attention = nn.Linear(64, self.last_layer(self.features(dummy_data)).shape[-1])
        self.feature_attention = nn.Linear(64, args.feature_dim)

    def forward(self, rotation, position, velocity, style_label, content_label, compute_grad):
        input_data = torch.cat((position, velocity), dim=-1).permute(0, 2, 1)
        if compute_grad: input_data.requires_grad_()
        features = self.last_layer(self.features(input_data))

        attention_input = torch.cat((style_label, content_label), dim=-1).mean(dim=1)
        attention_features = self.attention_features(attention_input)
        temporal_attention = self.temporal_attention(attention_features)
        feature_attention = self.feature_attention(attention_features)

        combined_features = (features * feature_attention.unsqueeze(-1)).sum(dim=1)
        final_score = (temporal_attention * combined_features).sum(dim=-1)
        grad = None
        if compute_grad:
            batch_size = final_score.shape[0]
            grad = autograd.grad(outputs=final_score.mean(),
                                 inputs=input_data,
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]
            grad = (grad ** 2).sum() / batch_size
        return final_score, grad


class RecurrentStylization(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        self.generator = Generator(args, dim_dict)
        self.discriminator = Discriminator(args, dim_dict)

    def forward_gen(self, rotation, position, velocity, content, contact, input_style, transferred_style,
                    test_time=False):
        return self.generator(rotation, position, velocity, content, contact, input_style, transferred_style,
                              test_time=test_time)

    def forward_dis(self, rotation, position, velocity, style_label, content_label, compute_grad=False):
        return self.discriminator(rotation, position, velocity, style_label, content_label, compute_grad=compute_grad)

    def forward(self, rotation, position, velocity, content, contact, root, input_style, transferred_style):
        generated_motion = self.forward_gen(rotation, position, velocity, content, contact, input_style,
                                            transferred_style)
        score = \
            self.forward_dis(generated_motion["rotation"], generated_motion["position"], generated_motion["velocity"],
                             transferred_style, content)[0]
        return generated_motion, score


class ContentClassification(nn.Module):
    def __init__(self, args, dim_dict):
        super().__init__()
        layers = []
        current_size = dim_dict["rotation"] + dim_dict["position"] + dim_dict["velocity"]
        for i in range(args.classifier_layer_num - 1):
            sub_layer = []
            sub_layer.append(conv_layer(5, current_size, (current_size // 4) * 2))
            if i < 5:
                sub_layer.append((nn.InstanceNorm1d((current_size // 4) * 2)))
            else:
                sub_layer.append(nn.BatchNorm1d((current_size // 4) * 2))
            sub_layer.append(nn.LeakyReLU())
            layers.append(nn.Sequential(*sub_layer))
            current_size = (current_size // 4) * 2
        self.feature_layers = nn.ModuleList(layers)
        self.classification_layers = conv_layer(5, current_size, args.content_num)

    def forward(self, rotation, position, velocity):
        data_input = torch.cat((rotation, position, velocity), dim=-1).permute(0, 2, 1)
        feature_output = {}
        x = data_input
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            feature_output["feature_{}".format(i)] = x
        output = self.classification_layers(x)
        return output.mean(dim=-1), feature_output
