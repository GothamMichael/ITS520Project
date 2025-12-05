import torch

from nio_recipe_full import (
    train_model,
    nio_optimize,
    INGREDIENTS,
    PROPERTIES,
    DEVICE,
    N_INPUTS,
)

def make_clamps(device):
    """
    Clamp ranges must match the physics generator:
    temp:    150–500
    weight:  50–600
    moisture:0.05–0.75
    umami/sweet/sour/spice: 0–10
    fat:     0–40
    carbs:   0–100
    protein: 0–50
    """
    clamp_min = torch.tensor([[
        150.0, 50.0, 0.05,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]], device=device)

    clamp_max = torch.tensor([[
        500.0, 600.0, 0.75,
        10.0, 10.0, 10.0, 10.0,
        40.0, 100.0, 50.0
    ]], device=device)

    return clamp_min, clamp_max


def run_scenario(
    name,
    model,
    stats,
    target,
    mask,
    overshoot,
    weights,
    clamp_min,
    clamp_max,
    max_steps=5000,
):
    """
    Run one NIO constraint setup and print what happens.
    """
    print(f"\n=== Scenario: {name} ===")
    target = target.to(DEVICE)
    overshoot = overshoot.to(DEVICE)
    mask = mask.to(torch.int32)
    weights = weights.to(torch.float32)

    final_x, final_y = nio_optimize(
        model=model,
        stats=stats,
        target_output_not_scaled=target,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        constraint_mask=mask,
        overshoot_margin_not_scaled=overshoot,
        constraint_weights=weights,
        alpha=0.9,
        max_steps=max_steps,
        lr=0.03,
        verbose=True,
    )

    if final_x is None:
        print("Result: NO FEASIBLE RECIPE FOUND under these constraints.")
        return

    x = final_x[0].cpu().numpy()
    y = final_y[0].cpu().numpy()

    print("Result: feasible recipe found.")
    print("\nInputs (ingredients):")
    for i, name_in in enumerate(INGREDIENTS):
        print(f"  {name_in:10s} = {x[i]:8.3f}")

    print("\nOutputs (properties):")
    for i, prop in enumerate(PROPERTIES):
        print(f"  {prop:10s} = {y[i]:8.3f}")


if __name__ == "__main__":
    # 1) Train the model once using Mike's physics data
    model, stats = train_model(
        n_epochs=200,
        N=5000,
        batch_size=128,
        lr=1e-3,
    )
    model.to(DEVICE)

    clamp_min, clamp_max = make_clamps(DEVICE)

    # PROPERTIES order: ["cook_time", "flavor", "calories"]

    # -----------------------------
    # Scenario 1: “Normal dinner”
    #   cook_time around 45–75 min
    #   flavor at least 0.5
    #   calories around 500–800
    # -----------------------------
    target_1 = torch.tensor([[60.0, 0.8, 650.0]], dtype=torch.float32)
    overshoot_1 = torch.tensor([20.0, 0.5, 200.0], dtype=torch.float32)
    mask_1 = torch.tensor([3, 2, 3], dtype=torch.int32)   # cook_time range, flavor >=, calories range
    weights_1 = torch.tensor([1.0, 3.0, 1.0], dtype=torch.float32)

    run_scenario(
        "Normal dinner",
        model,
        stats,
        target_1,
        mask_1,
        overshoot_1,
        weights_1,
        clamp_min,
        clamp_max,
    )

    # -----------------------------
    # Scenario 2: “Quick snack”
    #   cook_time <= 30 min
    #   flavor >= 0.8
    #   calories between 200–500
    #   (may or may not be feasible, which is good for demo)
    # -----------------------------
    target_2 = torch.tensor([[25.0, 1.0, 350.0]], dtype=torch.float32)
    overshoot_2 = torch.tensor([15.0, 0.5, 150.0], dtype=torch.float32)
    mask_2 = torch.tensor([3, 2, 3], dtype=torch.int32)
    weights_2 = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)

    run_scenario(
        "Quick snack",
        model,
        stats,
        target_2,
        mask_2,
        overshoot_2,
        weights_2,
        clamp_min,
        clamp_max,
    )

    # -----------------------------
    # Scenario 3: “Unrealistic diet fantasy”
    #   Basically impossible on purpose:
    #   5-minute cook, super high flavor, 100 calories
    # -----------------------------
    target_3 = torch.tensor([[5.0, 2.0, 100.0]], dtype=torch.float32)
    overshoot_3 = torch.tensor([3.0, 0.2, 20.0], dtype=torch.float32)
    mask_3 = torch.tensor([1, 2, 3], dtype=torch.int32)   # hard equality on cook_time
    weights_3 = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)

    run_scenario(
        "Unrealistic diet fantasy",
        model,
        stats,
        target_3,
        mask_3,
        overshoot_3,
        weights_3,
        clamp_min,
        clamp_max,
    )
    # -----------------------------
    # Scenario 4: "Cook-time only (easy mode)"
    #   Only constrain cook_time to be in a wide band around 300.
    #   Ignore flavor and calories completely.
    # -----------------------------
    target_4 = torch.tensor([[300.0, 0.0, 0.0]], dtype=torch.float32)
    overshoot_4 = torch.tensor([200.0, 0.0, 0.0], dtype=torch.float32)
    mask_4 = torch.tensor([3, 0, 0], dtype=torch.int32)   # cook_time in [100, 500], ignore others
    weights_4 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    run_scenario(
        "Cook-time only (easy mode)",
        model,
        stats,
        target_4,
        mask_4,
        overshoot_4,
        weights_4,
        clamp_min,
        clamp_max,
    )
