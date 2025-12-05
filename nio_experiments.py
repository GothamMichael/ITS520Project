from nio_recipe_full import (
    train_model,
    nio_optimize,
    INGREDIENTS,
    PROPERTIES,
    DEVICE
)
import torch

if __name__ == "__main__":
    # 1) Train the model using Mike's generator + loader
    model, stats = train_model(
        n_epochs=200,
        N=5000,
        batch_size=128,
        lr=1e-3,
    )

    # 2) Clamp ranges (same as Mike’s)
    clamp_min = torch.tensor([[
        150., 50., 0.05,
        0., 0., 0., 0.,
        0., 0., 0.
    ]], device=DEVICE)

    clamp_max = torch.tensor([[
        500., 600., 0.75,
        10., 10., 10., 10.,
        40., 100., 50.
    ]], device=DEVICE)

    print("Model ready. You can now begin NIO scenarios.")
