# Detecção Multimodal de Deepfakes com Small Language Models

**TCC — Engenharia Elétrica | UFES | 2026**
**Autor:** Gabryel Fonseca Batista
**Orientador:** Prof. Dr. Bruno Légora
**Coorientador:** Gabriel Braga Ladislau

## Descrição

Proposta e validação de uma arquitetura multimodal eficiente para detecção de imagens
sintéticas geradas por modelos de difusão. O sistema integra extração de características
visuais com a capacidade de raciocínio lógico de Small Language Models (SLMs),
ajustados via QLoRA (Quantized LoRA), viabilizando execução em hardware com recursos
limitados.

## Dataset

**SID_Set** — Social media Image Detection Set (Huang et al., 2025)
- 210k imagens de treino + 30k de validação
- 3 classes: REAL, SYNTHETIC (FLUX), TAMPERED (inpainting)
- Disponível em `saberzl/SID_Set` no HuggingFace

O dataset é baixado automaticamente na primeira execução e salvo em `dataset/sid_set/`.

## Ambiente

Requer [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/gabryelfbatista/deepfake-detection-tcc.git
cd deepfake-detection-tcc
uv sync
```

## Treinamento

```bash
# Modelo principal
uv run python src/train/experiment_01_slm_multimodal.py

# Baseline comparativo
uv run python src/train/experiment_01_cnn_baseline.py
```

Hiperparâmetros em `configs/slm_config_experiment_01.yaml` e `configs/cnn_config_experiment_01.yaml`.

## Avaliação

```bash
uv run python src/evaluate.py          # ambos os modelos
uv run python src/evaluate.py --modelo slm
uv run python src/evaluate.py --modelo cnn
```

Resultados salvos em `experiments/results_summary.csv`.

## Referências Principais

- Hu et al. (2021) — LoRA: Low-Rank Adaptation of Large Language Models
- Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs
- Huang et al. (2025) — SIDA: Social Media Image Deepfake Detection
- Wang et al. (2020) — CNN-Generated Images are Surprisingly Easy to Spot
