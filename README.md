# terramind-1-base

Deploy do **[ibm-esa-geospatial/TerraMind-1.0-base](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)** no Replicate via Cog. Foundation model multimodal any-to-any para Earth Observation (IBM + ESA + Forschungszentrum Jülich).

## Modelo
- **Tipo**: Backbone encoder multimodal
- **Pré-treinamento**: 500B tokens de 9M amostras multimodais (dataset TerraMesh)
- **Modalidades**: S2L2A, S2L1C, S1GRD, S1RTC, DEM, **RGB**
- **Saída**: Embeddings/features (não classificação direta)
- **Licença**: Apache 2.0
- **Paper**: [arxiv 2504.11171](https://arxiv.org/abs/2504.11171)

## ⚠️ Importante — é um BACKBONE, não um app

Este modelo retorna **embeddings** (vetores de features), não classificações ou descrições.
Use para:
- Similaridade entre imagens de satélite
- Treinar classificadores por cima (fine-tuning)
- Pesquisa em foundation models geoespaciais

Para tarefas específicas (canopy height, biomassa, detecção de inundação), use os modelos task-specific da família IBM Granite Geospatial.

## API

### Inputs

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `image` | Path | obrigatório | Imagem RGB (jpg/png) |
| `image_size` | int | 224 | Resolução (64-1024, múltiplo de 16 recomendado) |
| `return_format` | string | "summary" | `summary` (preview + stats) ou `full` (vetor completo) |

### Output (summary)

```json
{
  "embedding_dim": 768,
  "image_size": 224,
  "mean": -0.012,
  "std": 0.387,
  "min": -1.84,
  "max": 2.11,
  "embedding_preview": [-0.12, 0.34, ...],
  "embedding_norm": 18.42
}
```

### Output (full)

Inclui o array completo `embedding: [...]` (768-1024 valores).

## Hardware
- **gpu-t4** (16 GB) — modelo cabe com folga
- Inferência: ~1-3 s por imagem
