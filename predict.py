"""Predict.py mínimo de debug — só pra ver se container boota."""
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/src/hf-cache"

# Log MÓDULO carregado (antes de qualquer import pesado)
print(f"[module] Carregando predict.py at {time.time()}", flush=True)
sys.stdout.flush()

import torch
print(f"[module] torch {torch.__version__} loaded, cuda={torch.cuda.is_available()}", flush=True)

from cog import BasePredictor, Input, Path
print(f"[module] cog loaded", flush=True)

WEIGHTS_DIR = "/src/hf-cache"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === at t={t0}", flush=True)
        sys.stdout.flush()

        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            entries = sorted(os.listdir(WEIGHTS_DIR))
            print(f"[setup] /src/hf-cache entries: {entries}", flush=True)
            for e in entries:
                sub = os.path.join(WEIGHTS_DIR, e)
                if os.path.isdir(sub):
                    print(f"[setup]   {e}/: {sorted(os.listdir(sub))[:10]}", flush=True)
        except Exception as e:
            print(f"[setup] ERR listing: {e}", flush=True)

        print(f"[setup] CUDA: available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}", flush=True)

        # Tentando importar terratorch com timing
        print(f"[setup] Importing terratorch... (t={time.time()-t0:.1f}s)", flush=True)
        try:
            t_imp = time.time()
            from terratorch import BACKBONE_REGISTRY
            print(f"[setup] terratorch imported in {time.time()-t_imp:.1f}s", flush=True)
        except Exception as e:
            print(f"[setup] terratorch import FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            # Não falha o setup — apenas marca pra predict() reportar erro
            self.terratorch_error = str(e)
            self.model = None
            return

        # Tentando construir modelo
        print(f"[setup] Building model... (t={time.time()-t0:.1f}s)", flush=True)
        try:
            t_build = time.time()
            self.model = BACKBONE_REGISTRY.build(
                "terramind_v1_base",
                pretrained=True,
                modalities=["RGB"],
            )
            self.model = self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            print(f"[setup] Model built in {time.time()-t_build:.1f}s", flush=True)
        except Exception as e:
            print(f"[setup] Model build FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            self.terratorch_error = f"build failed: {e}"
            self.model = None
            return

        self.terratorch_error = None
        print(f"[setup] DONE in {time.time()-t0:.1f}s", flush=True)

    def predict(
        self,
        image: Path = Input(description="Imagem RGB ou TIF."),
        image_size: int = Input(default=224, ge=64, le=1024),
    ) -> str:
        if self.model is None:
            return json.dumps({
                "error": "Model failed to load during setup",
                "details": getattr(self, "terratorch_error", "unknown"),
            })

        import numpy as np
        from PIL import Image as PILImage

        pil = PILImage.open(image).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        device = next(self.model.parameters()).device
        tensor = tensor.to(device, dtype=torch.float32)

        with torch.no_grad():
            try:
                output = self.model({"RGB": tensor})
            except Exception:
                output = self.model(tensor)

        # Get features
        if isinstance(output, dict):
            feat = next(iter(output.values()))
        elif isinstance(output, (list, tuple)):
            feat = output[0]
        else:
            feat = output

        if feat.dim() == 3:
            feat = feat.mean(dim=1)

        v = feat.squeeze(0).cpu().numpy()
        return json.dumps({
            "embedding_dim": int(v.shape[-1]) if v.ndim >= 1 else 0,
            "image_size": image_size,
            "norm": float(np.linalg.norm(v)),
            "preview": v.flatten()[:16].tolist(),
        })
