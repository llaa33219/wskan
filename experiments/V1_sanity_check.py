"""V1 sanity check: fit a 1D multi-frequency target with WaveletStateKAN.

Static mode only. Reports loss trajectory, final RMSE, and saves a
qualitative fit plot if matplotlib is available.

Run:  python experiments/V1_sanity_check.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.V1_WSKAN import WaveletStateKAN


def target(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * torch.pi * x) + 0.5 * torch.cos(5 * torch.pi * x)


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    x = torch.linspace(-1, 1, 512, device=device).unsqueeze(-1)
    y = target(x)

    model = WaveletStateKAN([1, 8, 1], n_states=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(1, 2001):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"step {step:4d}  mse {loss.item():.6f}")

    with torch.no_grad():
        pred = model(x)
        rmse = (pred - y).pow(2).mean().sqrt().item()
    print(f"final RMSE: {rmse:.6f}")

    # Qualitative check: per-point residuals at a few probe inputs.
    probes = torch.tensor([[-0.75], [-0.25], [0.0], [0.33], [0.8]], device=device)
    with torch.no_grad():
        p = model(probes)
    for xi, pi in zip(probes[:, 0].tolist(), p[:, 0].tolist()):
        print(f"  x={xi:+.2f}  target={target(torch.tensor(xi)).item():+.4f}  pred={pi:+.4f}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(x.cpu(), y.cpu(), label="target")
        plt.plot(x.cpu(), pred.cpu(), "--", label="WSKAN V1")
        plt.legend()
        plt.tight_layout()
        out = Path(__file__).resolve().parent / "V1_sanity_check.png"
        plt.savefig(out, dpi=120)
        print(f"plot saved: {out}")
    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
