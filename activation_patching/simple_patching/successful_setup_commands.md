# Successful Setup Commands (from terminal 2)

These are the commands that completed successfully from your recent setup session.

## SSH and directory setup

```bash
ssh ubuntu@129.213.16.180
mkdir -p ~/simple_patch_31_prompts
cd ~/simple_patch_31_prompts
```

## SCP uploads (in order run)

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" "C:\Users\soohw\Simple_Patching\Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting\activation_patching\simple_patching\simple_patching_without_BHCs.py" ubuntu@129.213.16.180:~/simple_patch_31_prompts/
scp -i "$env:USERPROFILE\.ssh\id_ed25519" "C:\Users\soohw\Simple_Patching\Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting\activation_patching\simple_patching\requirements.txt" ubuntu@129.213.16.180:~/simple_patch_31_prompts/
```

## Virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
python -V
```

## Packaging tools upgrade

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Hugging Face token (recommended before model download)

```bash
export HF_TOKEN="hf_xxx_your_token_here"
```

## Dependency installation

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick checks you ran

```bash
ls
```

