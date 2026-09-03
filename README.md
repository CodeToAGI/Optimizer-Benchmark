# CodeToAGI — Deep Learning Series  
## Episode 19 Challenge: Optimizer Benchmark

**Episode title:** Optimizers Deep Dive — SGD, Adam, AdamW & When to Use Each  
**Module:** 5 — Optimizers & Advanced Training  
**Challenge file:** `ep19_optimizer_benchmark.py`

---

### What you will do

1. Take an EP18-style MLP (BatchNorm + Dropout + OneCycleLR).
2. Train it three times with identical architecture and data:
   - **SGD** (`lr=0.01`, `momentum=0.9`, Nesterov)
   - **Adam** (`lr=1e-3`)
   - **AdamW** (`lr=1e-3`, `weight_decay=1e-4`)
3. Record validation accuracy and wall-clock time for each run.
4. Plot the loss curves and the learning-rate schedule.
5. Answer:
   - Which optimizer converged fastest?
   - Which reached the highest final validation accuracy?
6. **Post your results table in the YouTube comments** of Episode 19.

---

### Quick start

```bash
# (optional) create a clean environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision matplotlib tqdm

python ep19_optimizer_benchmark.py
