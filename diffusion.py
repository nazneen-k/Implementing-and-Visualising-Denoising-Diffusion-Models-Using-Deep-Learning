"""
DDPM Core Module — Mathematical Implementation on 5×5 Images
=============================================================
Implements Denoising Diffusion Probabilistic Models (Ho et al., 2020)
on 5×5 single-channel images for step-by-step mathematical analysis.

Key formulas implemented:
  Forward:  q(x_t | x_0) = N(x_t; sqrt(ā_t)*x_0, (1-ā_t)*I)
  Reverse:  p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), β_t*I)
  Loss:     L = || ε - ε_θ(x_t, t) ||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------------------------------
# DIFFUSION PARAMETERS
# -------------------------------------------------------
T = 50  # Number of diffusion timesteps (50 for proper demonstration)
betas = torch.linspace(0.0001, 0.02, T)  # Linear noise schedule β_t
alphas = 1 - betas                        # α_t = 1 - β_t
alpha_hat = torch.cumprod(alphas, dim=0)  # ā_t = ∏_{s=1}^{t} α_s


# -------------------------------------------------------
# FORWARD DIFFUSION  q(x_t | x_0)
# -------------------------------------------------------
def forward_diffusion(x0, t):
    """
    Apply forward diffusion to get noisy image x_t from clean image x_0.

    Formula: x_t = √(ā_t) · x_0 + √(1 - ā_t) · ε,  where ε ~ N(0, I)

    Args:
        x0: Clean image tensor of shape (1, 1, 5, 5)
        t:  Timestep (int or tensor)

    Returns:
        x_t:   Noisy image at timestep t
        noise: The noise ε that was added
    """
    t = int(t)
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_hat[t]) * x0 + torch.sqrt(1 - alpha_hat[t]) * noise
    return x_t, noise


# -------------------------------------------------------
# REVERSE STEP  p_θ(x_{t-1} | x_t)
# -------------------------------------------------------
def reverse_step(x_t, t, pred_noise):
    """
    Perform one reverse diffusion step: compute x_{t-1} from x_t.

    Formula:
        μ_θ = (1/√α_t) · (x_t - (1-α_t)/√(1-ā_t) · ε_θ)
        x_{t-1} = μ_θ + σ_t · z,  where z ~ N(0,I), σ_t = √β_t

    At t=0, no noise is added (deterministic final step).

    Args:
        x_t:        Current noisy image
        t:          Current timestep (tensor)
        pred_noise: Model-predicted noise ε_θ(x_t, t)

    Returns:
        x_prev: Denoised image x_{t-1}
    """
    t = t.item()

    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)
    beta_t = betas[t].view(-1, 1, 1, 1)

    # Compute mean μ_θ
    mean = (1 / torch.sqrt(alpha_t)) * (
        x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        x_prev = mean + sigma * noise
    else:
        x_prev = mean  # No noise at final step

    return x_prev


# -------------------------------------------------------
# RECONSTRUCTION (x_0 prediction from x_t)
# -------------------------------------------------------
def reconstruct(x_t, pred_noise, t):
    """
    Predict x_0 directly from x_t using the predicted noise.

    Formula: x̂_0 = (1/√ā_t) · (x_t - √(1-ā_t) · ε_θ)

    This is a single-step reconstruction (not iterative denoising).
    """
    t = int(t)
    return (1 / torch.sqrt(alpha_hat[t])) * \
           (x_t - torch.sqrt(1 - alpha_hat[t]) * pred_noise)


# -------------------------------------------------------
# SINUSOIDAL TIME EMBEDDING
# -------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar timestep t to a high-dimensional embedding using
    sinusoidal positional encoding (Vaswani et al., 2017).

    This allows the model to distinguish different noise levels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb


# -------------------------------------------------------
# DENOISER MODEL  ε_θ(x_t, t)
# -------------------------------------------------------
class SimpleDenoiser(nn.Module):
    """
    A simple CNN-based noise predictor for 5×5 images.

    Architecture:
      - Sinusoidal time embedding (dim=32) → Linear → reshape to (1,5,5)
      - Time embedding is added to input image
      - 3-layer CNN: Conv(1→16) → ReLU → Conv(16→16) → ReLU → Conv(16→1)

    The model predicts the noise ε that was added at timestep t.
    """
    def __init__(self):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(32)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        self.fc = nn.Linear(32, 25)  # 25 = 5×5 pixels

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.fc(t_emb)
        t_emb = t_emb.view(-1, 1, 5, 5)
        x = x + t_emb  # Add time conditioning
        return self.net(x)


# -------------------------------------------------------
# QUALITY METRICS
# -------------------------------------------------------
def compute_psnr(x, y):
    """
    Peak Signal-to-Noise Ratio.

    Formula: PSNR = 10 · log₁₀(1 / MSE)

    Assumes pixel range [0, 1], so MAX_I = 1.
    """
    mse = F.mse_loss(x, y)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(1 / mse)


def compute_ssim(x, y):
    """
    Structural Similarity Index (simplified global version).

    For 5×5 images, the standard 11×11 Gaussian windowed SSIM
    cannot be applied. We compute global statistics instead.

    Constants (for dynamic range L=1):
        C1 = (K1·L)² = (0.01·1)² = 0.0001
        C2 = (K2·L)² = (0.03·1)² = 0.0009

    Formula:
        SSIM = (2μ_x·μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    sigma_x = torch.var(x)
    sigma_y = torch.var(y)
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y))

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim


# -------------------------------------------------------
# MODEL INSTANCE & OPTIMIZER
# -------------------------------------------------------
model = SimpleDenoiser()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def reset_model():
    """Reinitialize model weights and optimizer for new training."""
    global model, optimizer
    model = SimpleDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------------------------------------------
# SAMPLING (Full generation from pure noise)
# -------------------------------------------------------
def sample(model_instance, shape=(1, 1, 5, 5)):
    """
    Generate a new image by running the full reverse process
    starting from pure Gaussian noise x_T ~ N(0, I).

    Returns:
        x:     Final generated image
        steps: List of intermediate images at each reverse step
    """
    model_instance.eval()
    x = torch.randn(shape)
    steps = []

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t])
        with torch.no_grad():
            pred_noise = model_instance(x, t_tensor)
        x = reverse_step(x, t_tensor, pred_noise)
        steps.append(x.detach().clone())

    return x, steps