import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import VGGEncoder
from models.decoder import Decoder
from models.WCT import whiten_and_color
from utils import calc_mean_std, calc_covariance_matrix


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def process_features(self, features):
        mean, std = calc_mean_std(features)
        normalized = (features - mean) / std
        out = self.relu(self.conv1(normalized))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

    def generate_decoder_input(self, content_features, style_features):
        # Process features
        F_c = self.process_features(content_features)
        F_s = self.process_features(style_features)
        style_covariance = calc_covariance_matrix(F_s)
        content_mean, _ = calc_mean_std(F_c)
        b, c, _, _ = F_c.size()
        F_c_flat = F_c.view(b, c, -1)
        transformed = torch.bmm(style_covariance, F_c_flat).view_as(F_c)
        transformed = self.conv3(transformed)
        decoder_input = transformed + content_mean
        return decoder_input

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def generate(self, content_images, style_images, alpha=1.0, beta=1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        decoder_input = self.generate_decoder_input(content_features, style_features)
        t = alpha * decoder_input + (1 - alpha) * content_features
        _, content_temp_1, content_temp_2, _ = self.vgg_encoder(content_images, output_last_feature=False)
        _, style_temp_1, style_temp_2, _ = self.vgg_encoder(style_images, output_last_feature=False)
        mixed_feature_1 = whiten_and_color(content_temp_1, style_temp_1, beta=beta)
        mixed_feature_2 = whiten_and_color(content_temp_2, style_temp_2, beta=beta)
        out = self.decoder(t, mixed_feature_1, mixed_feature_2)
        return out

    def forward(self, content_images, style_images, alpha=1.0, lam=10, beta=1.0, cycle_lambda=10.0, id_lambda=5.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        decoder_input = self.generate_decoder_input(content_features, style_features)
        t = alpha * decoder_input + (1 - alpha) * content_features

        _, content_temp_1, content_temp_2, _ = self.vgg_encoder(content_images, output_last_feature=False)
        _, style_temp_1, style_temp_2, _ = self.vgg_encoder(style_images, output_last_feature=False)

        mixed_feature_1 = whiten_and_color(content_temp_1, style_temp_1)
        mixed_feature_2 = whiten_and_color(content_temp_2, style_temp_2)

        out = self.decoder(t, mixed_feature_1, mixed_feature_2)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(output_features, t)
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        
        regenerated_content_features = self.vgg_encoder(out, output_last_feature=True)
        regenerated_decoder_input = self.generate_decoder_input(regenerated_content_features, content_features)
        regenerated_t = alpha * regenerated_decoder_input + (1 - alpha) * regenerated_content_features
        _, regenerated_content_temp_1, regenerated_content_temp_2, _ = self.vgg_encoder(out, output_last_feature=False)
        regenerated_mixed_feature_1 = whiten_and_color(regenerated_content_temp_1, content_temp_1)
        regenerated_mixed_feature_2 = whiten_and_color(regenerated_content_temp_2, content_temp_2)
        regenerated_out = self.decoder(regenerated_t, regenerated_mixed_feature_1, regenerated_mixed_feature_2)
        loss_cycle = F.mse_loss(content_images, regenerated_out)

        content_decoder_input = self.generate_decoder_input(content_features, content_features)
        content_t = alpha * content_decoder_input + (1 - alpha) * content_features
        content_mixed_feature_1 = whiten_and_color(content_temp_1, content_temp_1)
        content_mixed_feature_2 = whiten_and_color(content_temp_2, content_temp_2)
        content_out = self.decoder(content_t, content_mixed_feature_1, content_mixed_feature_2)
        style_decoder_input = self.generate_decoder_input(style_features, style_features)
        style_t = alpha * style_decoder_input + (1 - alpha) * style_features
        style_mixed_feature_1 = whiten_and_color(style_temp_1, style_temp_1)
        style_mixed_feature_2 = whiten_and_color(style_temp_2, style_temp_2)
        style_out = self.decoder(style_t, style_mixed_feature_1, style_mixed_feature_2)
        loss_id = F.mse_loss(content_images, content_out) + F.mse_loss(style_images, style_out)

        loss = loss_c + lam * loss_s + cycle_lambda * loss_cycle + id_lambda * loss_id
        return loss
