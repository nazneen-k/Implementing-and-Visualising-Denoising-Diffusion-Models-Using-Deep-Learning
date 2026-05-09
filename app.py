"""
DDPM Interactive Mathematical Demo — 5×5 Images
=================================================
Streamlit application demonstrating the step-by-step mathematics
of Denoising Diffusion Probabilistic Models on 5×5 pixel images.

Sections:
  1. Input Image Selection
  2. Noise Schedule Analysis
  3. Model Training with Loss Analysis
  4. Forward Diffusion with Math
  5. Noise Prediction Analysis
  6. Reverse Diffusion (Step-by-Step)
  7. Image Generation from Pure Noise
  8. Quality Metrics
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from diffusion import (
    forward_diffusion, reverse_step, reconstruct, reset_model, sample,
    model, optimizer, T, betas, alphas, alpha_hat,
    compute_psnr, compute_ssim
)

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(page_title="DDPM Math Demo", layout="wide")

# -------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------
def display_matrix(arr, caption="", fmt=".4f"):
    """Display a 5×5 numpy array as a formatted table."""
    # Pre-format numbers as strings so they display correctly
    formatted = [[f"{arr[i][j]:{fmt}}" for j in range(5)] for i in range(5)]
    df = pd.DataFrame(
        formatted,
        columns=[f"c{j}" for j in range(5)],
        index=[f"r{i}" for i in range(5)]
    )
    if caption:
        st.caption(caption)
    st.table(df)


def display_image(tensor, caption="", width=200):
    """Display a tensor image upscaled 40× for visibility."""
    img = tensor.squeeze().detach().numpy()
    img = np.clip(img, 0, 1)
    img = np.kron(img, np.ones((40, 40)))
    st.image(img, caption=caption, width=width)


def normalize_for_display(arr):
    """Normalize array to [0,1] range for display."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# -------------------------------------------
# SESSION STATE
# -------------------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False
if "prev_option" not in st.session_state:
    st.session_state.prev_option = None
if "losses" not in st.session_state:
    st.session_state.losses = []
if "per_t_losses" not in st.session_state:
    st.session_state.per_t_losses = {}
if "prev_epochs" not in st.session_state:
    st.session_state.prev_epochs = None

# -------------------------------------------
# TITLE
# -------------------------------------------
st.title("🧠 Denoising Diffusion Probabilistic Model (DDPM)")
st.markdown("### Step-by-Step Mathematical Analysis on 5×5 Images")
st.markdown("---")

# ===============================
# SECTION 1: INPUT IMAGE
# ===============================
st.header("📌 1. Input Image (5×5)")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    option = st.selectbox(
        "Choose Pattern",
        ["Chessboard", "Diagonal", "Vertical Lines"]
    )

# Reset training when pattern changes
if st.session_state.prev_option != option:
    st.session_state.trained = False
    st.session_state.prev_option = option
    st.session_state.losses = []
    st.session_state.per_t_losses = {}
    # Clear analysis results
    for key in ["recon", "rev_steps", "generated", "gen_start", "noise_analysis"]:
        if key in st.session_state:
            del st.session_state[key]
    reset_model()

# Generate image
if option == "Chessboard":
    x0 = torch.zeros((1, 1, 5, 5))
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:
                x0[0, 0, i, j] = 1.0
elif option == "Diagonal":
    x0 = torch.zeros((1, 1, 5, 5))
    for i in range(5):
        x0[0, 0, i, i] = 1.0
elif option == "Vertical Lines":
    x0 = torch.zeros((1, 1, 5, 5))
    for j in range(5):
        if j % 2 == 0:
            x0[0, 0, :, j] = 1.0

with col2:
    display_image(x0, f"Original {option} (upscaled)")

with col3:
    st.markdown("**Raw 5×5 Pixel Values (x₀):**")
    display_matrix(x0.squeeze().numpy(), "", fmt=".1f")

st.markdown("---")

# ===============================
# SECTION 2: NOISE SCHEDULE
# ===============================
st.header("📌 2. Noise Schedule Analysis")

st.latex(r"\beta_t \in [\beta_1, \beta_T], \quad \alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s")

t_vals = np.arange(T)

col1, col2 = st.columns(2)

with col1:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(t_vals, betas.numpy(), 'b-o', markersize=2)
    axes[0].set_title("Beta_t (Noise Variance)")
    axes[0].set_xlabel("Timestep t")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_vals, alphas.numpy(), 'g-o', markersize=2)
    axes[1].set_title("Alpha_t = 1 - Beta_t")
    axes[1].set_xlabel("Timestep t")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_vals, alpha_hat.numpy(), 'r-o', markersize=2)
    axes[2].set_title("Alpha_bar_t (Cumulative)")
    axes[2].set_xlabel("Timestep t")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))

    signal_coeff = torch.sqrt(alpha_hat).numpy()
    noise_coeff = torch.sqrt(1 - alpha_hat).numpy()

    axes2[0].plot(t_vals, signal_coeff, 'b-o', ms=2, label='sqrt(alpha_bar_t) (signal)')
    axes2[0].plot(t_vals, noise_coeff, 'r-o', ms=2, label='sqrt(1 - alpha_bar_t) (noise)')
    axes2[0].set_title("Signal vs Noise Coefficients")
    axes2[0].set_xlabel("Timestep t")
    axes2[0].legend(fontsize=7)
    axes2[0].grid(True, alpha=0.3)

    snr = alpha_hat / (1 - alpha_hat)
    axes2[1].plot(t_vals, snr.numpy(), 'm-o', markersize=2)
    axes2[1].set_title("SNR = alpha_bar_t / (1 - alpha_bar_t)")
    axes2[1].set_xlabel("Timestep t")
    axes2[1].set_yscale('log')
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.info(
    f"**Schedule:** T={T}, β₁={betas[0].item():.6f}, β_T={betas[-1].item():.6f}, "
    f"ā_T={alpha_hat[-1].item():.6f}"
)

st.markdown("---")

# ===============================
# SECTION 3: TRAINING
# ===============================
st.header("📌 3. Model Training")

st.latex(r"\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]")

col_t1, col_t2 = st.columns([1, 3])
with col_t1:
    num_epochs = st.slider("Training Epochs", 500, 5000, 2000, step=500)

# Reset training when epoch count changes
if st.session_state.prev_epochs != num_epochs:
    if st.session_state.prev_epochs is not None:
        st.session_state.trained = False
        st.session_state.losses = []
        st.session_state.per_t_losses = {}
        for key in ["recon", "rev_steps", "generated", "gen_start", "noise_analysis"]:
            if key in st.session_state:
                del st.session_state[key]
        reset_model()
    st.session_state.prev_epochs = num_epochs

if not st.session_state.trained:
    torch.manual_seed(42)

    epoch_avg_losses = []
    per_t_losses = {t_step: [] for t_step in range(T)}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(num_epochs):
        # Train on ALL timesteps each epoch for consistent learning signal
        epoch_loss_sum = 0.0
        for t_step in range(T):
            t_train = torch.tensor([t_step])
            x_t, noise = forward_diffusion(x0, t_train)
            pred = model(x_t, t_train)

            loss = torch.nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            per_t_losses[t_step].append(loss.item())

        avg_loss = epoch_loss_sum / T
        epoch_avg_losses.append(avg_loss)

        if epoch % max(1, num_epochs // 20) == 0:
            progress_bar.progress(min(epoch / num_epochs, 1.0))
            status_text.text(f"Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

    progress_bar.progress(1.0)
    status_text.text(f"✅ Training complete! Final Avg Loss: {epoch_avg_losses[-1]:.6f}")

    st.session_state.trained = True
    st.session_state.losses = epoch_avg_losses
    st.session_state.per_t_losses = per_t_losses

if st.session_state.trained and st.session_state.losses:
    losses = st.session_state.losses

    # Compute early vs late loss for comparison
    n_compare = max(1, len(losses) // 10)
    early_loss = np.mean(losses[:n_compare])
    late_loss = np.mean(losses[-n_compare:])
    reduction_pct = ((early_loss - late_loss) / early_loss) * 100 if early_loss > 1e-8 else 0

    # Display summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Initial Avg Loss", f"{early_loss:.6f}")
    m2.metric("Final Avg Loss", f"{late_loss:.6f}")
    m3.metric("Loss Reduction", f"{reduction_pct:.1f}%",
              delta=f"-{reduction_pct:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))

        # Raw epoch-average losses (faint background)
        ax.plot(losses, alpha=0.25, linewidth=0.8, color='cornflowerblue',
                label='Per-Epoch Avg')

        # Exponential Moving Average for a clear smooth trend
        ema_alpha = min(0.05, 10.0 / len(losses))
        ema = [losses[0]]
        for l in losses[1:]:
            ema.append(ema_alpha * l + (1 - ema_alpha) * ema[-1])
        ax.plot(ema, color='#e74c3c', linewidth=2.5, label='Smoothed Trend (EMA)')

        # Mark start and end
        ax.scatter([0], [ema[0]], color='#e74c3c', s=50, zorder=5)
        ax.scatter([len(ema)-1], [ema[-1]], color='#27ae60', s=50, zorder=5)
        ax.annotate(f'{ema[0]:.4f}', (0, ema[0]), textcoords="offset points",
                    xytext=(10, 8), fontsize=8, color='#e74c3c', fontweight='bold')
        ax.annotate(f'{ema[-1]:.4f}', (len(ema)-1, ema[-1]),
                    textcoords="offset points", xytext=(-60, 8), fontsize=8,
                    color='#27ae60', fontweight='bold')

        ax.set_title("Training Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (avg over all timesteps)")
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        # Left: Average loss per timestep
        avg_losses = []
        for t_idx in range(T):
            losses_t = st.session_state.per_t_losses.get(t_idx, [])
            avg_losses.append(np.mean(losses_t) if losses_t else 0)
        axes[0].bar(range(T), avg_losses, color='steelblue', alpha=0.7)
        axes[0].set_title("Avg Loss per Timestep")
        axes[0].set_xlabel("Timestep t")
        axes[0].set_ylabel("Avg MSE Loss")
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right: Loss in phases (early vs mid vs late training)
        n_phases = min(5, len(losses))
        phase_size = len(losses) // n_phases
        phase_avgs = []
        phase_labels = []
        for i in range(n_phases):
            start = i * phase_size
            end = start + phase_size if i < n_phases - 1 else len(losses)
            phase_avgs.append(np.mean(losses[start:end]))
            phase_labels.append(f"E{start+1}-{end}")
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_phases))
        axes[1].bar(phase_labels, phase_avgs, color=colors, edgecolor='gray',
                    linewidth=0.5)
        axes[1].set_title("Loss by Training Phase")
        axes[1].set_ylabel("Avg MSE Loss")
        axes[1].tick_params(axis='x', labelsize=7, rotation=30)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown(
        "**Insight:** Higher timesteps (more noise) are generally harder to predict. "
        "The per-timestep loss shows which noise levels the model struggles with. "
        "The phase chart shows how loss decreases as training progresses."
    )

st.markdown("---")

# ===============================
# SECTION 4: FORWARD DIFFUSION
# ===============================
st.header("📌 4. Forward Diffusion (Adding Noise)")

st.latex(r"x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})")

t = st.slider("Select Timestep t", 0, T - 1, T // 4)

torch.manual_seed(123)
xt, noise_added = forward_diffusion(x0, t)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original x₀")
    display_image(x0, "x₀")
    display_matrix(x0.squeeze().numpy(), "x₀ values", fmt=".2f")

with col2:
    st.subheader(f"Noise ε (t={t})")
    noise_np = noise_added.squeeze().detach().numpy()
    noise_disp = normalize_for_display(noise_np)
    st.image(np.kron(noise_disp, np.ones((40, 40))),
             caption="ε ~ N(0, I)", width=200)
    display_matrix(noise_np, "ε values", fmt=".4f")

with col3:
    st.subheader(f"Noisy Image x_{t}")
    display_image(xt, f"x_{t} at t={t}")
    display_matrix(xt.squeeze().detach().numpy(), f"x_{t} values", fmt=".4f")

# Calculation details
st.markdown("#### 📐 Calculation Details")
sqrt_ahat = torch.sqrt(alpha_hat[t]).item()
sqrt_1m_ahat = torch.sqrt(1 - alpha_hat[t]).item()

calc_col1, calc_col2 = st.columns(2)
with calc_col1:
    st.latex(rf"\bar{{\alpha}}_{{{t}}} = {alpha_hat[t].item():.6f}")
    st.latex(rf"\sqrt{{\bar{{\alpha}}_{{{t}}}}} = {sqrt_ahat:.6f} \quad \text{{(signal coefficient)}}")
    st.latex(rf"\sqrt{{1 - \bar{{\alpha}}_{{{t}}}}} = {sqrt_1m_ahat:.6f} \quad \text{{(noise coefficient)}}")
with calc_col2:
    st.latex(rf"x_{{{t}}} = {sqrt_ahat:.4f} \times x_0 + {sqrt_1m_ahat:.4f} \times \epsilon")
    st.info(f"**Signal retention:** {sqrt_ahat*100:.2f}%  |  **Noise level:** {sqrt_1m_ahat*100:.2f}%")

st.markdown("---")

# ===============================
# SECTION 5: NOISE PREDICTION
# ===============================
st.header("📌 5. Noise Prediction Analysis")

st.latex(r"\hat{\epsilon}_\theta(x_t, t) \approx \epsilon")
st.latex(r"\mathcal{L} = \| \epsilon - \hat{\epsilon}_\theta(x_t, t) \|^2")

if st.button("🔍 Analyze Noise Prediction"):
    model.eval()
    with torch.no_grad():
        torch.manual_seed(456)
        x_t_a, true_noise = forward_diffusion(x0, t)
        pred_noise_a = model(x_t_a, torch.tensor([t]))
        noise_error = (true_noise - pred_noise_a).abs()
        mse_val = torch.nn.functional.mse_loss(pred_noise_a, true_noise)

    st.session_state.noise_analysis = {
        'true_noise': true_noise,
        'pred_noise': pred_noise_a,
        'error': noise_error,
        'mse': mse_val.item(),
        'x_t': x_t_a,
        't': t
    }
    model.train()

if "noise_analysis" in st.session_state:
    na = st.session_state.noise_analysis

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("True Noise ε")
        display_matrix(na['true_noise'].squeeze().detach().numpy(),
                       "True ε", fmt=".4f")
    with col2:
        st.subheader("Predicted ε̂_θ")
        display_matrix(na['pred_noise'].squeeze().detach().numpy(),
                       "Predicted ε̂", fmt=".4f")
    with col3:
        st.subheader("|ε − ε̂|  Error")
        display_matrix(na['error'].squeeze().detach().numpy(),
                       "Absolute error", fmt=".4f")

    st.metric("Prediction MSE", f"{na['mse']:.6f}")

    # Reconstruction
    st.markdown("#### Reconstruction from predicted noise:")
    st.latex(r"\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \hat{\epsilon}_\theta \right)")

    x_recon = reconstruct(na['x_t'], na['pred_noise'], na['t'])

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("**Original x₀:**")
        display_matrix(x0.squeeze().numpy(), "", fmt=".2f")
    with rc2:
        st.markdown("**Reconstructed x̂₀:**")
        display_matrix(x_recon.squeeze().detach().numpy(), "", fmt=".4f")
    with rc3:
        psnr_val = compute_psnr(x0, x_recon).item()
        ssim_val = compute_ssim(x0, x_recon).item()
        st.metric("PSNR", f"{psnr_val:.2f} dB")
        st.metric("SSIM", f"{ssim_val:.4f}")

st.markdown("---")

# ===============================
# SECTION 6: REVERSE PROCESS
# ===============================
st.header("📌 6. Reverse Diffusion (Step-by-Step Denoising)")

st.latex(r"\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)")
st.latex(r"x_{t-1} = \mu_\theta + \sigma_t \cdot z, \quad \sigma_t = \sqrt{\beta_t}, \quad z \sim \mathcal{N}(0, \mathbf{I})")

if st.button("▶️ Run Full Reverse Process"):
    model.eval()
    torch.manual_seed(789)
    x_T, _ = forward_diffusion(x0, T - 1)
    x = x_T.clone()

    steps = []
    with torch.no_grad():
        for t_step in reversed(range(T)):
            t_tensor = torch.tensor([t_step])
            pred_noise = model(x, t_tensor)
            x = reverse_step(x, t_tensor, pred_noise)
            steps.append(x.squeeze().detach().numpy())

    st.session_state.rev_steps = steps
    st.session_state.x_T = x_T
    model.train()

if "rev_steps" in st.session_state:
    num_show = min(6, T)
    gap = max(1, len(st.session_state.rev_steps) // num_show)

    cols = st.columns(num_show + 1)

    with cols[0]:
        img_xT = st.session_state.x_T.squeeze().detach().numpy()
        img_xT_norm = normalize_for_display(img_xT)
        st.image(np.kron(img_xT_norm, np.ones((40, 40))),
                 caption=f"x_T (pure noise)", use_container_width=True)

    for i in range(num_show):
        idx = min(i * gap, len(st.session_state.rev_steps) - 1)
        with cols[i + 1]:
            img = np.clip(st.session_state.rev_steps[idx], 0, 1)
            st.image(np.kron(img, np.ones((40, 40))),
                     caption=f"Step {idx+1}/{T}",
                     use_container_width=True)

    # Compare final vs original
    st.markdown("#### Final Result Comparison")
    fc1, fc2 = st.columns(2)
    with fc1:
        display_matrix(x0.squeeze().numpy(), "Original x₀", fmt=".2f")
    with fc2:
        display_matrix(st.session_state.rev_steps[-1],
                       "Reconstructed", fmt=".4f")

st.markdown("---")

# ===============================
# SECTION 7: GENERATION
# ===============================
st.header("📌 7. Image Generation (DDPM Sampling)")

st.latex(r"x_T \sim \mathcal{N}(0, \mathbf{I}) \quad \rightarrow \quad x_{T-1} \rightarrow \cdots \rightarrow x_0")
st.markdown(
    "Generate a **completely new** image by starting from pure Gaussian noise "
    "and running the full reverse process for all T steps."
)

gen_seed = st.number_input("Random Seed for Generation", value=42, step=1)

if st.button("🎲 Generate New Image from Pure Noise"):
    model.eval()
    torch.manual_seed(int(gen_seed))
    generated, gen_steps = sample(model)
    st.session_state.generated = generated
    st.session_state.gen_steps = gen_steps
    st.session_state.gen_start = torch.randn(1, 1, 5, 5)  # for display
    torch.manual_seed(int(gen_seed))
    st.session_state.gen_start = torch.randn(1, 1, 5, 5)
    model.train()

if "generated" in st.session_state:
    # Show progression
    st.markdown("#### Generation Progression (Noise → Image)")
    gen_steps = st.session_state.gen_steps
    num_show = min(8, len(gen_steps))
    gap = max(1, len(gen_steps) // num_show)

    cols = st.columns(num_show + 1)

    with cols[0]:
        start_img = st.session_state.gen_start.squeeze().detach().numpy()
        start_disp = normalize_for_display(start_img)
        st.image(np.kron(start_disp, np.ones((40, 40))),
                 caption="x_T (noise)", use_container_width=True)

    for i in range(num_show):
        idx = min(i * gap, len(gen_steps) - 1)
        with cols[i + 1]:
            img = gen_steps[idx].squeeze().detach().numpy()
            img_disp = np.clip(img, 0, 1)
            st.image(np.kron(img_disp, np.ones((40, 40))),
                     caption=f"Step {idx+1}/{T}",
                     use_container_width=True)

    # Final generated image details
    st.markdown("#### Generated Image Details")
    gc1, gc2 = st.columns(2)
    with gc1:
        display_image(st.session_state.generated, "Generated Image")
        display_matrix(st.session_state.generated.squeeze().detach().numpy(),
                       "Generated pixel values", fmt=".4f")
    with gc2:
        st.markdown("**Comparison with Original:**")
        display_image(x0, f"Original {option}")
        display_matrix(x0.squeeze().numpy(), "Original pixel values", fmt=".2f")

    # Pixel difference
    diff = (st.session_state.generated.squeeze().detach() - x0.squeeze()).abs().numpy()
    st.markdown("**Absolute Pixel Difference |Generated − Original|:**")
    display_matrix(diff, "Pixel-wise error", fmt=".4f")

st.markdown("---")

# ===============================
# SECTION 8: QUALITY METRICS
# ===============================
st.header("📌 8. Quality Metrics Summary")

st.latex(r"\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{1}{\text{MSE}}\right)")
st.latex(r"\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}")

if "generated" in st.session_state:
    gen = st.session_state.generated
    psnr_gen = compute_psnr(x0, gen).item()
    ssim_gen = compute_ssim(x0, gen).item()
    mse_gen = torch.nn.functional.mse_loss(x0, gen).item()

    qm1, qm2, qm3 = st.columns(3)
    qm1.metric("MSE", f"{mse_gen:.6f}")
    qm2.metric("PSNR", f"{psnr_gen:.2f} dB")
    qm3.metric("SSIM", f"{ssim_gen:.4f}")

    # Quality interpretation
    if psnr_gen > 30:
        st.success("🟢 **Excellent quality** — Generated image closely matches the original.")
    elif psnr_gen > 20:
        st.warning("🟡 **Moderate quality** — Visible differences from the original.")
    else:
        st.error("🔴 **Low quality** — Significant deviation from the original pattern.")

    st.markdown(
        "**Interpretation:**\n"
        "- **MSE** measures average squared pixel error (lower = better)\n"
        "- **PSNR** measures signal quality in dB (higher = better, >30 dB is excellent)\n"
        "- **SSIM** measures structural similarity (closer to 1.0 = better)"
    )
else:
    st.info("Generate an image in Section 7 to see quality metrics here.")

st.markdown("---")
st.caption("DDPM Interactive Demo — M.Tech Project")
