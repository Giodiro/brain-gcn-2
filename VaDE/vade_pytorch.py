import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib as plt
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime



def time_str():
    now = datetime.now()
    return now.strftime("[%m-%d %H:%M:%S]")

def fig_to_array(fig, close=False):
    ax = fig.gca()
    plt.setp([ax.get_xticklines() + ax.get_yticklines() +
              ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)
    matplotlib.rcParams['text.antialiased'] = False

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.array(fig.canvas.renderer._renderer)

    if close:
        plt.close(fig)

    return data


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def build_network(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)



class VaDE(nn.Module):
    def __init__(self, input_dim, encoder_dims, z_dim, decoder_dims, n_centroids, no_cuda=False):
        super(VaDE, self).__init__()

        ## Build the network
        self.encoder = build_network([input_dim] + encoder_dims, activation="relu", dropout=0.0)
        self.decoder = build_network(decoder_dims, activation="relu", dropout=0.0)
        self.dec_first_stage = nn.Sequential(*[nn.Linear(z_dim, decoder_dims[0]), nn.ReLU()])
        self.decoder = nn.Sequential(*[self.decoder, nn.Linear(decoder_dims[-1], input_dim), nn.Sigmoid()])

        self._enc_mu = nn.Linear(encoder_dims[-1], z_dim)
        self._enc_logvar = nn.Linear(encoder_dims[-1], z_dim)

        # theta is the categorical distribution
        self.theta_p = nn.Parameter(torch.ones(n_centroids) / n_centroids)
        # u is the means of the gaussian mixture components
        self.u_p = nn.Parameter(torch.zeros(z_dim, n_centroids))
        # lambda is the (diagonal) covariances of GMM components
        self.lambda_p = nn.Parameter(torch.ones(z_dim, n_centroids))

        ## Initialize parameters
        self.lr = 1.0e-3
        self.pretrain_lr = self.lr
        self.n_centroids = n_centroids
        self.lr_decay = 0.999
        self.summary_images = 10
        self.weight_decay = 1.0e-10

        use_cuda = torch.cuda.is_available() and not no_cuda
        self.device = "cuda" if use_cuda else "cpu"
        self.to(self.device)


    def initialize_gmm(self, dataloader):
        from sklearn.mixture import GaussianMixture

        print(f"{time_str()} Fitting sklearn GMM with {self.n_centroids} components...")

        self.is_pretraining = False
        all_z = []
        for inputs, labels in dataloader:
            z, x_reconstructed, mu, logvar = self.forward(inputs.to(self.device))
            all_z.append(z.data.cpu().numpy())

        all_z = np.concatenate(all_z)
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type="diag")
        gmm.fit(all_z)

        self.u_p.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

        print(f"{time_str()} Finished initialization of GMM components.")

    def get_gamma(self, z):
        """
        Args:
            z : latent dimension of size [batch_size x z_dim]
            z_mean : latent mean of size [batch_size x z_dim]
            z_logvar : size [batch_size x z_dim]
        """

        # add the n_centroids dimension at end
        Z = z.unsqueeze(2).expand(z.size(0), z.size(1), self.n_centroids) # B x D x K

        # add the batch_size dimension at beginning
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size(0), self.u_p.size(0), self.u_p.size(1)) #
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size(0), self.lambda_p.size(0), self.lambda_p.size(1)) #
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size(0), self.theta_p.size(0)) #

        p_c_z = torch.exp(theta_tensor2.log() - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3) +
                                                           (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1.0e-10

        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma

    def loss(self, x, x_reconstructed, z, z_mean, z_logvar):

        # add the n_centroids dimension at end
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size(0), z_mean.size(1), self.n_centroids)
        z_logvar_t = z_logvar.unsqueeze(2).expand(z_logvar.size(0), z_logvar.size(1), self.n_centroids)

        # add the batch_size dimension at beginning
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size(0), self.u_p.size(0), self.u_p.size(1)) #
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size(0), self.lambda_p.size(0), self.lambda_p.size(1)) #
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size(0), self.theta_p.size(0)) #

        gamma = self.get_gamma(z)

        cross_entropy = -torch.sum(x * torch.log(torch.clamp(x_reconstructed, min=1.0e-10)) +
                        (1-x) * torch.log(torch.clamp(1 - x_reconstructed, min=1.0e-10)), dim=1)
        logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2 * math.pi) +
                                               torch.log(lambda_tensor3) +
                                               torch.exp(z_logvar_t)/lambda_tensor3 +
                                               (z_mean_t - u_tensor3)**2 / lambda_tensor3, dim=1), dim=1)
        qentropy = -0.5*torch.sum(1+z_logvar + math.log(2*math.pi), dim=1)
        logpc = -torch.sum(torch.log(theta_tensor2) * gamma, dim=1)
        logqcx = torch.sum(torch.log(gamma) * gamma, 1)

        loss = torch.mean(cross_entropy + logpzc + qentropy + logpc + logqcx)

        return loss

    def autoencoder_loss(self, x, x_reconstructed):
        cross_entropy = F.binary_cross_entropy(x_reconstructed, x)#, reduction="elementwise_mean")

        return cross_entropy

    def forward(self, x):
        x = x.to(self.device)
        if self.is_pretraining:
            h = self.encoder(x)
            x_reconstructed = self.decoder(h)
            return x_reconstructed

        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_logvar(h)
        z = self.reparametrize(mu, logvar)
        recon = self.decoder(self.dec_first_stage(z))

        return z, recon, mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)

        return mu

    def run_epoch(self, dataloader, is_training):
        if is_training:
            self.train()
        else:
            self.eval()

        tot_loss = []; Y_pred = []; Y_true = []

        for batch_i, (inputs, labels) in enumerate(dataloader):
            z, x_reconstructed, mu, logvar = self.forward(inputs)
            loss = self.loss(inputs, x_reconstructed, z, mu, logvar)

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tot_loss.append([loss.data.cpu().numpy()] * len(inputs))

            gamma = self.get_gamma(z).data.cpu().numpy()

            Y_pred.append(np.argmax(gamma, axis=1))
            Y_true.append(labels.cpu().numpy())

            self.global_step += len(inputs)

        if self.scheduler is not None:
            self.scheduler.step(epoch=self.epoch)


        tot_loss = np.concatenate(tot_loss)
        Y_pred = np.concatenate(Y_pred)
        Y_true = np.concatenate(Y_true)

        # Print epoch output to screen
        try:
            lr = self.scheduler.get_lr()[0]
        except AttributeError:
            lr = float(self.optimizer.param_groups[0]['lr'])

        epoch_desc = "train" if is_training else "validation"

        print((f"{time_str()} Epoch {self.epoch} - lr: {lr} - "
               f"{epoch_desc} loss: {np.mean(tot_loss)} "
               f"- accuracy: {cluster_acc(Y_pred, Y_true)}"))

        # Write tensorboard summaries
        self.summary.add_scalar(f"loss/{epoch_desc}", np.mean(tot_loss), self.global_step)

        if self.epoch % self.summary_images == 0:
            # Image summaries
            # 1. bar chart of the cluster priors
            priors = self.theta_p.data.cpu().numpy()
            fig, ax = plt.subplots()
            ax.bar(range(len(priors)), priors, 1/1.5, color="blue", align="center")
            ax.set_xlabel("Categorical priors")

            npfig = fig_to_array(fig, close=True)
            self.summary.add_image(f"cat_priors/{epoch_desc}", npfig, self.global_step)

    def pretrain(self, num_epochs, tr_loader):
        self.is_pretraining = True

        parameters = []
        for p in self.encoder.parameters():
            if p.requires_grad:
                parameters.append(p)
        for p in self.decoder.parameters():
            if p.requires_grad:
                parameters.append(p)
        optimizer = torch.optim.Adam(parameters, lr=self.pretrain_lr)

        for epoch in range(num_epochs):
            epoch_losses = []
            for batch_i, (inputs, labels) in enumerate(tr_loader):
                x_reconstructed = self.forward(inputs)
                loss = self.autoencoder_loss(inputs, x_reconstructed)
                epoch_losses.append(loss.data.cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"{time_str()} pretrain epoch {epoch}: loss {np.mean(epoch_losses):.4}")


    def fit(self, num_epochs, tr_loader, val_loader):
        self.is_pretraining = False
        self.global_step = 0

        parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr,
            weight_decay=self.weight_decay)

        self.scheduler = None
        if self.lr_decay is not None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.lr_decay)

        for self.epoch in range(num_epochs):
            self.run_epoch(tr_loader, is_training=True)
            self.run_epoch(val_loader, is_training=False)


    def sample(self):
        sample = torch.randn(64, self.z_dim).to(self.device)
        sample = self.decode(sample).cpu()
        save_image
