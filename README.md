# GARCH credit

Volatilidad condicional para crédito corporativo: OAS (EGARCH), EBP (GJR), VaR/ES por FHS, régimen, score de dislocación y overlay de exposición sobre pesos HRP.

Los umbrales viven en `config/`. El código no hardcodea knobs.

## Setup

Python ≥ 3.11. Para datos en vivo: `FRED_API_KEY` en el entorno.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Correr

```bash
python scripts/bootstrap_data.py          # universo FRED + ETF
python scripts/estimate_oas_panel.py      # EGARCH en las 5 series OAS
python scripts/estimate_ebp.py            # GJR en EBP mensual / diario
python scripts/run_fhs.py                 # VaR / ES (FHS-GJR)
python scripts/plot_dislocation.py --fit  # score histórico
python scripts/run_signal_backtest.py     # walk-forward vs 4 benchmarks
```

## Tests

```bash
pytest -m "not slow"     # ciclo diario
pytest -m blocking       # look-ahead + oráculos (gate de merge)
pytest -m slow           # simulaciones grandes (CI)
```

CI: cobertura ≥ 80% en `src/`, ≥ 95% en `src/risk/` y `src/models/`.

## Layout

| Ruta | Qué es |
|---|---|
| `src/data/` | Ingesta, validación, Chow-Lin / EBP |
| `src/models/` | OAS EGARCH, EBP GJR, régimen |
| `src/risk/` | FHS, backtests VaR/ES, overlay |
| `src/signals/` | Score de dislocación |
| `src/backtest/` | Walk-forward de la señal |
| `src/diagnostics/` | Gate econométrico (opt-in) |
| `config/` | YAML validado por pydantic |
