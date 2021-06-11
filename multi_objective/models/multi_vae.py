# Taken from https://github.com/swisscom/ai-research-mamo-framework/blob/master/models/params_multi_VAE.yaml and adapted (Apache license).
# Thanks!


"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml


class MultiVAE(nn.Module):
    """An implementation of a Multi Variational Auto Encoder model.

    VAE is an autoencoder whose encodings distribution is regularised during
    the training in order to ensure that its latent space has good properties allowing
    us to generate some new data. More info on VAEs
    https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

    Attributes:
        The values for all the parameters are stored in params_multi_VAE.yaml
        input_size: The dimension of the input as an Integer.
        output_size: The dimension of the output as an Integer.
        dropout: A float between 0 and 1 (default value is 0.5).
        no_latent_features: Number of latent features as an Integer
            Normally less than both input_size and output_size.
            Default value is 200
        encoder: Takes an input of size input_size and processes it to
            give the mean and log_variance (each of size no_latent_features).
            Default definition is nn.ModuleList(modules=[
                                     nn.Linear(input_size, 600),
                                     nn.Linear(600, 400)])
        decoder: Takes an input of size no_latent_features and
            processes it to give an output of size output_size.
            Default definition is nn.ModuleList(modules=[
                                     nn.Linear(200, 600),
                                     nn.Linear(600, output_size)])
        activation_function: Specify the activation function to be used
            Default value: tanh()
        params: Specify the path to the yaml file which has the parameters,
            or a dictionary containing the parameters.
            Default value: 'params_multi_VAE.yaml'
    """

    def __init__(self, vae_params, activation_function=torch.tanh, **kwargs):
        """Initialize the model"""
        super(MultiVAE, self).__init__()
        self.params = vae_params

        self.input_size = self.params['input_size']
        self.output_size = self.params['output_size']
        self.activation_function = activation_function
        self.dropout = nn.Dropout(self.params['dropout'])

        self.initialize_model()

    def _load_params(self, path):
        with open(path, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return(params)

    def initialize_model(self):
        """Initializes the model"""
        # create a model
        self.__create_model(self.input_size, self.output_size)

        # intialize weights
        self.__initialize_weights()

    def __create_model(self, input_size, output_size):
        """Create a model with the specified parameters

           Default values:
               enc1_out: 600
               ecn2_in: 600
               enc2_out: 400
               dec1_in: 200
               dec1_out: 600
               dec2_in: 600"""
        # create encoder
        self.encoder = nn.ModuleList(modules=[
            nn.Linear(self.input_size, self.params['enc1_out']),
            nn.Linear(self.params['enc2_in'], self.params['enc2_out'])
        ])

        # create decoder
        self.decoder = nn.ModuleList(modules=[
            nn.Linear(self.params['dec1_in'], self.params['dec1_out']),
            nn.Linear(self.params['dec2_in'], self.output_size)
        ])
        self.no_latent_features = self.params['no_latent_features']

    def forward(self, batch):
        """A single forward pass of the model

        Args:
            input: The input to the model as a tensor of batch_size X input_size

        Raises:
            NotImplementedError
        """
        input = batch['data']
        # encode
        mean, log_variance = self.encode(input)

        # compute latent features
        z = self.reparameterize(mean, log_variance)

        # decode
        return dict(logits=self.decode(z), mean=mean, log_variance=log_variance)

    def encode(self, input):
        """The encoding function for the  VAE

        Args:
            input: The input to the model as a tensor of batch_size X input_size

        Returns:
            mean and log_variance as tensors of length batch_size X no_latent_features
                each entry of the tensor being a float
        """
        # define result variables
        mean = None
        log_variance = None

        # save final layer
        final_layer = self.encoder[-1]

        # normalize input
        x = F.normalize(input)
        # dropout
        x = self.dropout(x)

        # go through encoder
        for layer in self.encoder:
            x = layer(x)
            if layer != final_layer:
                x = self.activation_function(x)
            else:
                # split mean and log_variance
                mean = x[:, :self.no_latent_features]
                log_variance = x[:, self.no_latent_features:]

        # return mean and log_variance tuple
        return mean, log_variance

    def reparameterize(self, mean, log_variance):
        """The reparamatrization trick

        https://towardsdatascience.com/reparameterization-trick-126062cfd3c3"""
        # if training add noise
        if self.training:
            std = torch.exp(1/2 * log_variance)
            epsilon = torch.randn_like(std)
            return epsilon.mul(std).add_(mean)
        else:
            return mean

    def decode(self, z):
        """The decoding function for the  VAE"""
        # save final layer
        final_layer = self.decoder[-1]

        # go through decoder
        x = z
        for layer in self.decoder:
            x = layer(x)
            if layer != final_layer:
                x = self.activation_function(x)
        return x

    def __initialize_weights(self):
        """Define initial values for weights

           Default values:
               norm_mean: 0.0
               norm_std: 0.001"""
        # Initialize encoder
        for layer in self.encoder:
            # Xavier Initialization for weights
            torch.nn.init.xavier_normal_(layer.weight.data)
            # Normal Initialization for Biases
            torch.nn.init.normal_(layer.bias.data, self.params['norm_mean'], self.params['norm_std'])

        # Initialize decoder
        for layer in self.decoder:
            # Xavier Initialization for weights
            torch.nn.init.xavier_normal_(layer.weight.data)
            # Normal Initialization for Biases
            torch.nn.init.normal_(layer.bias.data, self.params['norm_mean'], self.params['norm_std'])
    

    def change_input_dim(self, dim):
        self.encoder[0] = nn.Linear(dim, self.encoder[0].out_features)