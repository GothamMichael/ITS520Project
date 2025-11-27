# nio_recipe_full.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------------------
# Config / Device
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------
# Synthetic recipe generator
# ---------------------------
INGREDIENTS = [
    "flour", "sugar", "butter", "eggs", "milk", "cocoa", "yeast"
]
PROPERTIES = ["sweetness", "moisture", "density"]
N_INPUTS = len(INGREDIENTS)
N_OUTPUTS = len(PROPERTIES)


def generate_synthetic_data(N=5000):
    # Generate ingredient raw values in plausible ranges
    X = np.zeros((N, N_INPUTS), dtype=np.float32)
    X[:, 0] = np.random.uniform(100, 500, size=N)   # flour (g)
    X[:, 1] = np.random.uniform(10, 300, size=N)    # sugar (g)
    X[:, 2] = np.random.uniform(20, 200, size=N)    # butter (g)
    X[:, 3] = np.random.uniform(1, 5, size=N)       # eggs (count)
    X[:, 4] = np.random.uniform(50, 400, size=N)    # milk (mL)
    X[:, 5] = np.random.uniform(0, 100, size=N)     # cocoa (g)
    X[:, 6] = np.random.uniform(0, 12, size=N)      # yeast (g)

    # Compute normalized-ish properties (add small noise)
    flour, sugar, butter, eggs, milk, cocoa, yeast = X.T

    sweetness = (
        0.6 * (sugar / 300.0) +
        0.3 * (cocoa / 100.0) +
        0.1 * (butter / 200.0) +
        np.random.normal(0, 0.02, size=N)
    )

    moisture = (
        0.5 * (milk / 400.0) +
        0.2 * (butter / 200.0) +
        0.2 * (eggs / 5.0) +
        np.random.normal(0, 0.02, size=N)
    )

    density = (
        0.5 * (flour / 500.0) -
        0.3 * (yeast / 12.0) +
        0.2 * (eggs / 5.0) +
        np.random.normal(0, 0.02, size=N)
    )

    Y = np.vstack([sweetness, moisture, density]).T.astype(np.float32)
    return X.astype(np.float32), Y


# ---------------------------
# Data prep
# ---------------------------
def prepare_loaders(N=5000, batch_size=64):
    from synthetic_recipe_physics import load_physics_dataset_as_numpy
    X_np, Y_np = load_physics_dataset_as_numpy(N)
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, Y_np, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    # compute scaling stats on train set
    x_means = X_train_t.mean(dim=0)
    x_stds = X_train_t.std(dim=0) + 1e-6
    y_means = y_train_t.mean(dim=0)
    y_stds = y_train_t.std(dim=0) + 1e-6

    # Train loader uses scaled targets (so model can predict scaled then descaled)
    train_ds = TensorDataset(X_train_t, (y_train_t - y_means) / y_stds)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(X_test_t, (y_test_t - y_means) / y_stds)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    stats = {
        "x_means": x_means,
        "x_stds": x_stds,
        "y_means": y_means,
        "y_stds": y_stds,
        "X_test": X_test_t,
        "y_test": y_test_t,
    }
    return train_dl, test_dl, stats


# ---------------------------
# Model definitions
# ---------------------------
class ResidualNet(nn.Module):
    def __init__(self, x_means, x_stds, y_means, y_stds, dropout_rate=0.1):
        super().__init__()
        self.x_means = x_means
        self.x_stds = x_stds
        self.y_means = y_means
        self.y_stds = y_stds

        self.fc1 = nn.Linear(N_INPUTS, 64)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, 64)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, N_OUTPUTS)
        self.input_proj = nn.Linear(N_INPUTS, 64) if N_INPUTS != 64 else nn.Identity()

    def forward(self, x_raw):
        # x_raw is in real units; scale first
        x = (x_raw - self.x_means) / self.x_stds
        x0 = self.input_proj(x)
        x1 = self.drop1(self.act1(self.fc1(x)))
        x2 = self.drop2(self.act2(self.fc2(x1)))
        x_res = x2 + x0
        y_scaled = self.fc3(x_res)                     # model outputs scaled targets
        y_descaled = y_scaled * self.y_stds + self.y_means
        return y_descaled, y_scaled


# ---------------------------
# Training helper
# ---------------------------
def train_model(n_epochs=100, N=5000, batch_size=64, lr=1e-3):
    train_dl, test_dl, stats = prepare_loaders(N=N, batch_size=batch_size)

    model = ResidualNet(
        stats["x_means"], stats["x_stds"], stats["y_means"], stats["y_stds"]
    ).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(1, n_epochs + 1):
        running = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            _, y_scaled = model(xb)
            loss = loss_fn(y_scaled, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / (len(train_dl.dataset))
        if epoch % max(1, n_epochs // 10) == 0 or epoch == 1:
            print(f"[Train] Epoch {epoch}/{n_epochs} loss={epoch_loss:.6f}")

    # quick eval on test set (descaled)
    model.eval()
    with torch.no_grad():
        X_test = stats["X_test"]
        y_test = stats["y_test"]
        y_pred_descaled, _ = model(X_test)
        mse = ((y_pred_descaled - y_test) ** 2).mean().item()
    print(f"[Eval] Test MSE (descaled) = {mse:.6f}")

    return model, stats


# ---------------------------
# NIO: helpers for inference / optimization
# ---------------------------
def get_x_from_z_fn(clamp_min, clamp_max):
    """
    returns a function mapping z -> x in real units.
    We use sigmoid(z) to map R -> (0,1), then scale to [clamp_min, clamp_max].
    clamp_min/max should be torch tensors on device.
    """
    def get_x_from_z(z_param):
        s = torch.sigmoid(z_param)
        return clamp_min + (clamp_max - clamp_min) * s
    return get_x_from_z


def soft_box_penalty(x, lower, upper, strength=10.0):
    # squared hinge outside [lower, upper]
    loss_low = torch.relu(lower - x).pow(2).sum()
    loss_high = torch.relu(x - upper).pow(2).sum()
    return strength * (loss_low + loss_high)


def regularize_z(z, strength=1e-3):
    return strength * torch.sum(z ** 2)


def constraint_loss(y_pred_scaled, y_target_scaled, mask, overshoot_margin, weights, reg_weight=1e-4,
                    lower_bounds_scaled=None, upper_bounds_scaled=None):
    """
    y_pred_scaled: model outputs (scaled) shape [1, n_outputs]
    y_target_scaled: scaled target (same shape)
    mask: tensor of codes: 0=no constraint,1=eq,2=>=,3=range
    overshoot_margin: in scaled units (same size) or raw? We'll assume scaled here for simplicity.
    weights: per-output weights
    """
    loss = 0.0
    for i in range(y_pred_scaled.shape[1]):
        w = weights[i]
        if mask[i] == 1:  # equality
            loss = loss + w * (y_pred_scaled[0, i] - y_target_scaled[0, i]) ** 2
        elif mask[i] == 2:  # >= constraint (penalize undershoot, and large overshoot)
            loss = loss + w * torch.relu(y_target_scaled[0, i] - y_pred_scaled[0, i]) ** 2
            safe_upper = y_target_scaled[0, i] + overshoot_margin[i]
            loss = loss + w * torch.relu(y_pred_scaled[0, i] - safe_upper) ** 2
        elif mask[i] == 3:  # bounded [lower_bounds_scaled, upper_bounds_scaled]
            a = lower_bounds_scaled[0, i]
            b = upper_bounds_scaled[0, i]
            loss = loss + w * torch.relu(a - y_pred_scaled[0, i]) ** 2
            loss = loss + w * torch.relu(y_pred_scaled[0, i] - b) ** 2
    loss = loss + reg_weight * torch.sum(y_pred_scaled ** 2)
    return loss


def nio_optimize(
    model,
    stats,
    target_output_not_scaled,
    clamp_min,
    clamp_max,
    constraint_mask,
    overshoot_margin_not_scaled,
    constraint_weights,
    alpha=0.9,
    max_steps=20000,
    lr=0.03,
    z_init=None,
    verbose=False,
):
    """
    Optimize z (latent) so that model(x) meets the constraints.
    - model: frozen model (eval mode)
    - stats: dictionary with scaling stats
    - target_output_not_scaled: torch tensor shape [1, n_outputs] on DEVICE in real units
    - clamp_min/max: tensors shape [1, n_inputs] on DEVICE (real units)
    - constraint_mask: integer tensor shape [n_outputs]
    - overshoot_margin_not_scaled: tensor shape [n_outputs] (real units)
    - constraint_weights: tensor shape [n_outputs]
    Returns final_input (1,n_inputs) and final_output_descaled (1,n_outputs) or (None,None) if fail.
    """

    model = model.to(DEVICE)
    model.eval()

    # convert target and overshoot to scaled space using stats
    y_means = stats["y_means"]
    y_stds = stats["y_stds"]

    target_scaled = (target_output_not_scaled.to(DEVICE) - y_means) / y_stds
    overshoot_scaled = overshoot_margin_not_scaled.to(DEVICE) / y_stds

    # prepare bounds in scaled space for mask==3
    lower_bounds_glob_scaled = ((target_output_not_scaled.to(DEVICE) - (overshoot_margin_not_scaled * 0)) - y_means) / y_stds
    upper_bounds_glob_scaled = ((target_output_not_scaled.to(DEVICE) + overshoot_margin_not_scaled) - y_means) / y_stds

    # prepare get_x mapping
    clamp_min = clamp_min.to(DEVICE)
    clamp_max = clamp_max.to(DEVICE)
    get_x_from_z = get_x_from_z_fn(clamp_min, clamp_max)

    n_trials = 3
    for attempt in range(n_trials):
        # initialize z (logit of mid-range by default)
        if z_init is None:
            mid = 0.5 * (clamp_min + clamp_max)
            # initialize z so that sigmoid(z) near 0.5 -> x approx mid
            z0 = torch.logit(torch.full((1, N_INPUTS), 0.5, device=DEVICE) * 0.4 + 0.3)  # keep in stable range
            z = torch.nn.Parameter(z0.clone())
        else:
            z = torch.nn.Parameter(z_init.clone().to(DEVICE))

        optimizer_infer = optim.Adam([z], lr=lr)
        # optional scheduler removed for simplicity

        for step in range(max_steps):
            optimizer_infer.zero_grad()
            x_guess = get_x_from_z(z)                     # real units
            y_pred_descaled, y_pred_scaled = model(x_guess)

            loss_main = constraint_loss(
                y_pred_scaled,
                target_scaled,
                constraint_mask,
                overshoot_scaled,
                constraint_weights,
                reg_weight=1e-4,
                lower_bounds_scaled=lower_bounds_glob_scaled,
                upper_bounds_scaled=upper_bounds_glob_scaled,
            )

            loss_soft = soft_box_penalty(x_guess, clamp_min, clamp_max, strength=10.0)
            loss_z = regularize_z(z, strength=1e-3)

            loss = alpha * loss_main + (1 - alpha) * loss_soft + loss_z
            loss.backward()
            optimizer_infer.step()

            # light clamp of z to avoid extreme logits (optional)
            with torch.no_grad():
                z.clamp_(min=-10.0, max=10.0)

            if verbose and (step % 2000 == 0):
                print(f"[attempt {attempt+1}] step {step} loss {loss.item():.6f}")

        # evaluate final
        with torch.no_grad():
            final_input = get_x_from_z(z)
            final_output_descaled, final_output_scaled = model(final_input)

        # Check mask conditions in descaled space (since constraints were expressed in real units)
        violations = torch.zeros_like(final_output_descaled, dtype=torch.bool)
        # compute violations considering mask types: equality / >= / range
        for i in range(N_OUTPUTS):
            if constraint_mask[i] == 1:  # equality (allow small tol)
                tol = 1e-2 * (abs(target_output_not_scaled[0, i]) + 1.0)
                violations[0, i] = not (abs(final_output_descaled[0, i] - target_output_not_scaled[0, i]) <= tol)
            elif constraint_mask[i] == 2:  # >=
                lower = target_output_not_scaled[0, i]
                upper = target_output_not_scaled[0, i] + overshoot_margin_not_scaled[i]
                if final_output_descaled[0, i] < lower - 1e-6 or final_output_descaled[0, i] > upper + 1e-6:
                    violations[0, i] = True
            elif constraint_mask[i] == 3:  # within [lower_bounds_glob, upper_bounds_glob]
                lower = (lower_bounds_glob_scaled[0, i] * y_stds[i] + y_means[i])  # convert back, simpler to use not-scaled values
                upper = (upper_bounds_glob_scaled[0, i] * y_stds[i] + y_means[i])
                # but since lower_bounds_glob_scaled was derived from target in this script, we can compute directly:
                lower_real = (lower_bounds_glob_scaled[0, i] * y_stds[i] + y_means[i])
                upper_real = (upper_bounds_glob_scaled[0, i] * y_stds[i] + y_means[i])
                if final_output_descaled[0, i] < lower_real - 1e-6 or final_output_descaled[0, i] > upper_real + 1e-6:
                    violations[0, i] = True

        # If no violations and an extra domain check (e.g., o_fta safety), accept
        o_fta = final_output_descaled[0, 0].item()
        if not violations.any() and o_fta <= 2100:
            if verbose:
                print("Success after attempt", attempt + 1)
            return final_input.cpu(), final_output_descaled.cpu()

        if verbose:
            print(f"Attempt {attempt+1} failed; violations: {violations.any().item()}, o_fta={o_fta:.2f}")

    # no feasible solution found
    return None, None


# ---------------------------
# Example run: train + NIO optimize
# ---------------------------
if __name__ == "__main__":
    # 1) Train
    model, stats = train_model(n_epochs=200, N=5000, batch_size=128, lr=1e-3)

    # 2) Define clamp_min / clamp_max (example values; shape [1, N_INPUTS])
    # NOTE: use realistic recipe ranges (same units as generator above)
    clamp_min = torch.tensor([[100., 10., 20., 1., 50., 0., 0.]], device=DEVICE)
    clamp_max = torch.tensor([[500., 300., 200., 5., 400., 100., 12.]], device=DEVICE)

    # 3) Example target (in real units): sweetness (~0..1), moisture (~0..1), density (~0..1)
    # But our synthetic outputs are already in roughly 0..1 range; so choose targets in that range.
    target_output_not_scaled = torch.tensor([[0.85, 0.6, 0.35]], dtype=torch.float32)  # desired real-unit properties
    overshoot_margin_not_scaled = torch.tensor([0.05, 0.1, 0.1], dtype=torch.float32)     # allowable overshoot in real units
    # constraint mask: 2 = >=, 3 = bounded range, 1 = equality, 0 = no constraint
    constraint_mask = torch.tensor([2, 2, 2], dtype=torch.int32)  # example: require >= for all three
    constraint_weights = torch.tensor([1.0, 3.0, 1.0], dtype=torch.float32)

    # 4) Run NIO optimization
    final_x, final_y = nio_optimize(
        model=model,
        stats=stats,
        target_output_not_scaled=target_output_not_scaled,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        constraint_mask=constraint_mask,
        overshoot_margin_not_scaled=overshoot_margin_not_scaled,
        constraint_weights=constraint_weights,
        alpha=0.9,
        max_steps=5000,
        lr=0.03,
        verbose=True,
    )

    if final_x is not None:
        print("\nFound feasible input (ingredient amounts):")
        print({INGREDIENTS[i]: float(final_x[0, i]) for i in range(N_INPUTS)})
        print("Predicted properties:")
        print({PROPERTIES[i]: float(final_y[0, i]) for i in range(N_OUTPUTS)})
    else:
        print("No feasible input found.")

