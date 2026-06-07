import pandas as pd
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import csv
from datetime import datetime
from pathlib import Path
from config import MAX_EDGE_LENGTH_KM, N_LAYERS, N_QUBITS, SEED

Edge_Distance = ["e_ab", "e_bc", "e_cd", "e_da", "e_ac", "e_bd"]
rng = np.random.default_rng(seed=SEED)
num_steps = 10000


def load_real_data(valid_tuples_path: str) -> np.ndarray:
    df = pd.read_csv(valid_tuples_path)
    missing = [col for col in Edge_Distance if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {valid_tuples_path}: {missing}")
    real = df[Edge_Distance].to_numpy(dtype=np.float32)
    real = np.clip(real / MAX_EDGE_LENGTH_KM, 0.0, 1.0)
    #print(f"[INFO] Loaded real data from {valid_tuples_path} (shape: {real.shape}, missing columns: {missing})")
    return real


def sample_real_data(real_data: np.ndarray, num_samples: int) -> np.ndarray:
    idx = rng.choice(len(real_data), size=num_samples, replace=True)
    #print(f"[INFO] Sampled {num_samples} real data points (indices: {idx})")
    return real_data[idx], idx

class QGenerator(nn.Module):

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()

        self.num_qubits = N_QUBITS
        self.num_layers = N_LAYERS
        self.device = qml.device("default.qubit", wires=self.num_qubits)

        self.input_param = nn.Parameter(0.01 * torch.randn(self.num_layers * self.num_qubits * 3))

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(input, input_param):
            
            qml.AngleEmbedding(input, wires=range(self.num_qubits), rotation="Y")

            idx = 0
            for _ in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RX(input_param[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits):
                    qml.RY(input_param[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits):
                    qml.RZ(input_param[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits):
                   qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
                    

            return [qml.probs(wires=i) for i in range(self.num_qubits)]

        self.circuit = circuit
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        probs = self.circuit(input, self.input_param)
        return torch.stack([p[1] for p in probs], dim=-1).to(dtype=torch.float32)

class discriminator(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
"""    
def main():
    torch.manual_seed(SEED)
    real_data = load_real_data("valid_tuples.csv")
    loss_fn = nn.BCEWithLogitsLoss()
    qgen = QGenerator(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    disc = discriminator()
    d_optimizer = torch.optim.Adam(disc.parameters(), lr=0.03, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(qgen.parameters(), lr=0.001, betas=(0.5, 0.999))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/hybrid_v2_{timestamp}_{SEED}")
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "metrics.csv"

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "d_loss", "g_loss", 
                        "real_score_d_roh", "fake_score_d_roh", "fake_score_g_roh", 
                        "real_score_d", "fake_score_d", "fake_score_g",
                        "d_grad_norm",
                          "g_grad_norm", "fake_mean", "fake_std","real_idx",
                            "real_e_ab", "real_e_bc", "real_e_cd", "real_e_da", "real_e_ac", "real_e_bd",
"fake_e_ab", "fake_e_bc", "fake_e_cd", "fake_e_da", "fake_e_ac", "fake_e_bd"],
        )
        writer.writeheader()

    for steps in range(num_steps):
        # Train Disc
        input =  2.0 * torch.pi * torch.rand(N_QUBITS)
        fake_distances = qgen(input).unsqueeze(0)
        #real_batch = sample_real_data(real_data, num_samples=1), dtype=torch.float32
        real_np, real_idx = sample_real_data(real_data, num_samples=1)
        real_batch = torch.tensor(
           real_np,
            dtype=torch.float32
       )

        d_real = disc(real_batch)
        d_fake = disc(fake_distances.detach())
        real_loss = loss_fn(d_real, torch.ones_like(d_real))
        fake_loss = loss_fn(d_fake, torch.zeros_like(d_fake))
        d_loss = (real_loss + fake_loss)*0.5
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Gen
        g_fake_for_d = disc(fake_distances)
        g_loss = loss_fn(g_fake_for_d, torch.ones_like(g_fake_for_d))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            real_score_d_roh = d_real.mean().item()
            fake_score_d_roh = d_fake.mean().item()
            fake_score_g_roh = g_fake_for_d.mean().item()
            real_score = torch.sigmoid(d_real).mean().item()
            fake_score_d = torch.sigmoid(d_fake).mean().item()
            fake_score_g = torch.sigmoid(g_fake_for_d).mean().item()
            d_grad_norm = torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0).item()
            g_grad_norm = torch.nn.utils.clip_grad_norm_(qgen.parameters(), max_norm=10.0).item()
            fake_mean = fake_distances.mean().item()
            fake_std = fake_distances.std().item()
            fake_e_ab = fake_distances[0, 0].item()
            fake_e_bc = fake_distances[0, 1].item()
            fake_e_cd = fake_distances[0, 2].item()
            fake_e_da = fake_distances[0, 3].item()
            fake_e_ac = fake_distances[0, 4].item()
            fake_e_bd = fake_distances[0, 5].item()
            real_e_ab = real_batch[0, 0].item()
            real_e_bc = real_batch[0, 1].item()
            real_e_cd = real_batch[0, 2].item()
            real_e_da = real_batch[0, 3].item()
            real_e_ac = real_batch[0, 4].item()
            real_e_bd = real_batch[0, 5].item()
        
            metrics = {
                "step": steps + 1,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "real_score_d_roh": real_score_d_roh,
                "fake_score_d_roh": fake_score_d_roh,
                "fake_score_g_roh": fake_score_g_roh,
                "real_score_d": real_score,
                "fake_score_d": fake_score_d,
                "fake_score_g": fake_score_g,
                "d_grad_norm": d_grad_norm,
                "g_grad_norm": g_grad_norm,
                "fake_mean": fake_mean,
                "fake_std": fake_std,
                "real_idx": int(real_idx[0]),
                "real_e_ab": real_e_ab,
                "real_e_bc": real_e_bc,
                "real_e_cd": real_e_cd,
                "real_e_da": real_e_da,
                "real_e_ac": real_e_ac,
                "real_e_bd": real_e_bd,

                "fake_e_ab": fake_e_ab,
                "fake_e_bc": fake_e_bc,
                "fake_e_cd": fake_e_cd,
                "fake_e_da": fake_e_da,
                "fake_e_ac": fake_e_ac,
                "fake_e_bd": fake_e_bd,
            }
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)

        if (steps + 1) % 100 == 0:
            print(
                f"Step {steps+1}/{num_steps} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f} | "
                f"Real: {real_score:.3f} | "
                f"Fake-D: {fake_score_d:.3f} | "
                f"Fake-G: {fake_score_g:.3f}"
            )



"""
def main():
    torch.manual_seed(SEED)
    qgen = QGenerator(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    cdisc= discriminator()
    params = torch.randn(N_LAYERS * N_QUBITS * 3)
    for i in range(10):
        
        input = 2.0 * torch.pi * torch.rand(N_QUBITS)
        probs = qgen.circuit(input, params)
        sample = torch.stack([p[1] for p in probs]).to(dtype=torch.float32)
        valid = cdisc(sample)
        print(f"\n Disc: {valid.item():.4f}")
        sample2 = torch.stack([p[1] for p in probs]).detach().numpy()
        print(f"\n GEN Sample {i+1}")
        print(probs)
        print(pd.DataFrame([sample2], columns=Edge_Distance).to_string(index=False))


if __name__ == "__main__":
    main()