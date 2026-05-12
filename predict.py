import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Aponta cache HF pra pasta pré-populada na build (estrutura HF cache real)
os.environ["HF_HUB_CACHE"] = "/src/hf-cache"

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

WEIGHTS_DIR = "/src/hf-cache"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(WEIGHTS_DIR))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list WEIGHTS_DIR: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()}", flush=True)

        print(f"[setup] importing terratorch... (t={time.time()-t0:.1f}s)", flush=True)
        from terratorch import BACKBONE_REGISTRY
        self.BACKBONE_REGISTRY = BACKBONE_REGISTRY

        # Aponta cache local pro terratorch encontrar o checkpoint
        # O terratorch espera o arquivo TerraMind_v1_base.pt em algum diretório acessível.
        # Vamos usar TORCH_HOME ou similar.
        os.environ["TERRAMIND_CHECKPOINT_PATH"] = CHECKPOINT_PATH

        print(f"[setup] building terramind_v1_base backbone (RGB)... (t={time.time()-t0:.1f}s)", flush=True)
        # Carrega backbone com modalidade RGB (mais acessível pro user comum)
        self.model = BACKBONE_REGISTRY.build(
            "terramind_v1_base",
            pretrained=True,
            modalities=["RGB"],
        )
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.float32)
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def predict(
        self,
        image: Path = Input(description="Imagem RGB (jpg/png) — preferencialmente de satélite/aérea."),
        image_size: int = Input(
            description="Resolução de entrada (deve ser múltiplo de patch size, ex: 224, 256, 512).",
            default=224,
            ge=64,
            le=1024,
        ),
        return_format: str = Input(
            description="Formato do output: 'summary' (estatísticas + embedding compacto), 'full' (vetor completo).",
            default="summary",
            choices=["summary", "full"],
        ),
    ) -> str:
        device = next(self.model.parameters()).device

        # Carregar e pré-processar imagem
        pil = Image.open(image).convert("RGB")
        pil = pil.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0  # [H, W, 3] em [0, 1]
        # TerraMind backbone espera tensor [B, C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)

        # Backbone retorna features. O API exato depende da versão do terratorch.
        # Tentamos os formatos mais comuns.
        with torch.no_grad():
            try:
                # Multi-modal input: dict com modalidade
                output = self.model({"RGB": tensor})
            except Exception:
                output = self.model(tensor)

        # Normalizar output em features [B, dim]
        if isinstance(output, dict):
            # Tenta pegar 'features' ou 'last_hidden_state' ou primeiro tensor disponível
            for k in ["features", "last_hidden_state", "embeddings", "x"]:
                if k in output:
                    feat = output[k]
                    break
            else:
                feat = next(iter(output.values()))
        elif isinstance(output, (list, tuple)):
            feat = output[0]
        else:
            feat = output

        # Se for sequência de tokens, faz pooling
        if feat.dim() == 3:
            # [B, N_tokens, dim] -> [B, dim]
            feat = feat.mean(dim=1)

        feat_vec = feat.squeeze(0).cpu().numpy()
        result = {
            "embedding_dim": int(feat_vec.shape[-1]) if feat_vec.ndim >= 1 else 0,
            "image_size": image_size,
        }

        if return_format == "full":
            result["embedding"] = feat_vec.flatten().tolist()
        else:
            result["mean"] = float(feat_vec.mean())
            result["std"] = float(feat_vec.std())
            result["min"] = float(feat_vec.min())
            result["max"] = float(feat_vec.max())
            result["embedding_preview"] = feat_vec.flatten()[:32].tolist()
            result["embedding_norm"] = float(np.linalg.norm(feat_vec))

        return json.dumps(result, ensure_ascii=False)
