from __future__ import print_function

import abc
import logging
import random

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from utils.normalize import CIFARNORMALIZE
from model.cv.resnet_v2 import ResNet18



# ============================================================
# Basic blocks
# ============================================================


def conv3x3(
        in_planes,
        out_planes,
        stride=1):

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True
    )



def conv_init(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        init.xavier_uniform_(
            m.weight,
            gain=np.sqrt(2)
        )

        if m.bias is not None:

            init.constant_(
                m.bias,
                0
            )


    elif classname.find("BatchNorm") != -1:

        init.constant_(
            m.weight,
            1
        )

        init.constant_(
            m.bias,
            0
        )



class wide_basic(nn.Module):

    def __init__(
            self,
            in_planes,
            planes,
            dropout_rate,
            stride=1):

        super().__init__()


        self.bn1 = nn.BatchNorm2d(
            in_planes
        )


        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=True
        )


        self.dropout = nn.Dropout(
            p=dropout_rate
        )


        self.bn2 = nn.BatchNorm2d(
            planes
        )


        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )


        self.shortcut = nn.Sequential()


        if stride != 1 or in_planes != planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True
                )

            )



    def forward(self,x):

        out = self.dropout(
            self.conv1(
                F.relu(
                    self.bn1(x)
                )
            )
        )


        out = self.conv2(
            F.relu(
                self.bn2(out)
            )
        )


        out += self.shortcut(x)


        return out




class ResBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            mid_channels=None,
            bn=False):

        super().__init__()


        if mid_channels is None:

            mid_channels = out_channels



        layers = [

            nn.LeakyReLU(),

            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.LeakyReLU(),

            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

        ]


        if bn:

            layers.insert(
                2,
                nn.BatchNorm2d(
                    out_channels
                )
            )



        self.convs = nn.Sequential(
            *layers
        )



    def forward(self,x):

        return x + self.convs(x)




class AbstractAutoEncoder(nn.Module):

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def encode(self,x):
        pass


    @abc.abstractmethod
    def decode(self,z):
        pass


    @abc.abstractmethod
    def forward(self,x):
        pass
class FL_CVAE_cifar(AbstractAutoEncoder):

    def __init__(
            self,
            args,
            d,
            z,
            device,
            with_classifier=True,
            **kwargs):


        super().__init__()


        self.device = device


        # =========================
        # Noise / privacy enhance
        # =========================

        self.noise_mean = getattr(
            args,
            "VAE_mean",
            0
        )


        self.noise_std1 = getattr(
            args,
            "VAE_std1",
            0.01
        )


        self.noise_std2 = getattr(
            args,
            "VAE_std2",
            0.01
        )


        self.noise_type = getattr(
            args,
            "noise_type",
            "Gaussian"
        )


        self.privacy_clip = getattr(
            args,
            "privacy_clip",
            False
        )


        self.privacy_clip_norm = float(
            getattr(
                args,
                "privacy_clip_norm",
                1.0
            )
        )



        # =========================
        # Encoder
        # =========================


        in_channels = (

            1
            if args.dataset == "fmnist"
            else
            3

        )


        self.encoder_former = nn.Conv2d(

            in_channels,

            d // 2,

            kernel_size=4,

            stride=2,

            padding=1,

            bias=False

        )



        self.encoder = nn.Sequential(

            nn.BatchNorm2d(
                d // 2
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Conv2d(
                d // 2,
                d,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),


            nn.BatchNorm2d(
                d
            ),


            nn.ReLU(
                inplace=True
            ),


            ResBlock(
                d,
                d,
                bn=True
            ),


            nn.BatchNorm2d(
                d
            ),


            ResBlock(
                d,
                d,
                bn=True
            )

        )



        # =========================
        # Decoder
        # =========================


        self.decoder = nn.Sequential(

            ResBlock(
                d,
                d,
                bn=True
            ),


            nn.BatchNorm2d(
                d
            ),


            ResBlock(
                d,
                d,
                bn=True
            ),


            nn.BatchNorm2d(
                d
            ),


            nn.ConvTranspose2d(

                d,

                d // 2,

                kernel_size=4,

                stride=2,

                padding=1,

                bias=False

            ),


            nn.BatchNorm2d(
                d // 2
            ),


            nn.LeakyReLU(
                inplace=True
            )

        )



        self.decoder_last = nn.ConvTranspose2d(

            d // 2,

            in_channels,

            kernel_size=4,

            stride=2,

            padding=1,

            bias=False

        )



        self.xi_bn = nn.BatchNorm2d(
            in_channels
        )


        self.sigmoid = nn.Sigmoid()



        # =========================
        # Latent
        # =========================

        self.f = 8

        self.d = d

        self.z = z



        self.fc11 = nn.Linear(
            d * self.f ** 2,
            z
        )


        self.fc12 = nn.Linear(
            d * self.f ** 2,
            z
        )


        self.fc21 = nn.Linear(
            z,
            d * self.f ** 2
        )



        self.with_classifier = with_classifier



        if self.with_classifier:


            self.classifier = ResNet18(

                args=args,

                num_classes=args.num_classes,

                image_size=32,

                model_input_channels=args.model_input_channels

            )



    # =====================================================
    # Privacy helpers
    # =====================================================


    def _clip_sample_l2(
            self,
            data,
            clip_norm=1.0):


        flat = data.view(
            data.size(0),
            -1
        )


        norm = torch.norm(
            flat,
            p=2,
            dim=1,
            keepdim=True
        )


        scale = torch.clamp(

            float(clip_norm)
            /
            (norm + 1e-12),

            max=1.0

        )


        return (
            flat * scale
        ).view_as(data)



    def _add_noise(
            self,
            data,
            mean,
            std):


        size = data.size()



        if self.noise_type == "Gaussian":


            noise = torch.normal(

                mean=float(mean),

                std=float(std),

                size=size,

                device=data.device

            )


        elif self.noise_type == "Laplace":


            noise = torch.tensor(

                np.random.laplace(

                    loc=mean,

                    scale=std,

                    size=size

                ),

                dtype=data.dtype,

                device=data.device

            )


        else:

            raise ValueError(
                f"Unknown noise type {self.noise_type}"
            )


        return data + noise



    # =====================================================
    # VAE encoder decoder
    # =====================================================


    def encode(self,x):

        h = self.encoder(x)


        h = h.view(

            -1,

            self.d * self.f ** 2

        )


        return (

            h,

            self.fc11(h),

            self.fc12(h)

        )



    def reparameterize(
            self,
            mu,
            logvar):


        if self.training:


            std = torch.exp(
                0.5 * logvar
            )


            eps = torch.randn_like(
                std
            )


            return mu + eps * std


        else:

            return mu



    def decode(self,z):

        z = z.view(

            -1,

            self.d,

            self.f,

            self.f

        )


        return torch.tanh(
            self.decoder(z)
        )



    # =====================================================
    # Forward
    # =====================================================


    def forward(self,x):


        original_x = x



        h = self.encoder_former(
            x
        )


        _, mu, logvar = self.encode(
            h
        )



        hi = self.reparameterize(
            mu,
            logvar
        )


        hi_projected = self.fc21(
            hi
        )


        xi = self.decode(
            hi_projected
        )


        xi = self.decoder_last(
            xi
        )


        xi = self.xi_bn(
            xi
        )


        xi = self.sigmoid(
            xi
        )



        rx = (
            original_x
            -
            xi
        )



        if self.privacy_clip:


            rx = self._clip_sample_l2(

                rx,

                self.privacy_clip_norm

            )



        rx_noise1 = self._add_noise(

            torch.clone(rx),

            self.noise_mean,

            self.noise_std1

        )


        rx_noise2 = self._add_noise(

            torch.clone(rx),

            self.noise_mean,

            self.noise_std2

        )



        if self.with_classifier:


            data = torch.cat(

                (

                    rx_noise1,

                    rx_noise2,

                    x

                ),

                dim=0

            )


            out = self.classifier(
                data
            )


            return (

                out,

                hi,

                xi,

                mu,

                logvar,

                rx,

                rx_noise1,

                rx_noise2

            )



        else:


            return (

                None,

                hi,

                xi,

                mu,

                logvar,

                rx,

                rx_noise1,

                rx_noise2

            )



    # =====================================================
    # Classifier interface
    # =====================================================


    def classifier_test(
            self,
            data):


        if not self.with_classifier:

            raise RuntimeError(
                "Classifier is disabled"
            )


        return self.classifier(
            data
        )



    def get_classifier(self):

        return self.classifier