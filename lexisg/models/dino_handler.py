import argparse
import os
from contextlib import nullcontext

import torch 
from PIL import Image
import numpy as np
from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

# lexisg/models/dino_handler.py -> repo root is two levels up.
_CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")

class Dino:
    def  __init__(self, device=None): 
        print("Loading Dino")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_txt_model, self.tokenizer= self._load_img_txt_model()
        self.image_preprocess = make_classification_eval_transform()

    def _autocast_context(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.float)
        return nullcontext()

    def get_img_feat(self, images):
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        image_tensors = torch.stack(
            [self.image_preprocess(Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")) for img in images],
            dim=0,
        ).to(self.device)
        with self._autocast_context():
            with torch.no_grad():
                image_feat = self.img_txt_model.encode_image(image_tensors)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        return image_feat.detach().cpu().numpy()
    
    def get_img_txt_feat(self, images):
        image_tensors = torch.stack(
            [self.image_preprocess(Image.open(img)) for img in images], dim=0
        ).to(self.device)
        with self._autocast_context():
            with torch.no_grad():
                image_feat = self.img_txt_model.encode_image(image_tensors)
        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        return image_feat.detach().cpu().numpy()
    
    def get_txt_feat(self, texts, batch_size=32):
        all_feats = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokenized = self.tokenizer.tokenize(batch_texts).to(self.device)
            with torch.no_grad():
                with self._autocast_context():
                    text_feat = self.img_txt_model.encode_text(tokenized)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            all_feats.append(text_feat.detach().cpu())
            del tokenized, text_feat
            torch.cuda.empty_cache()
        return torch.cat(all_feats, dim=0).numpy()
    
    def get_similarity(self, feat1, feat2):
        return feat1 @ feat2.T

    @staticmethod
    def _existing_path(path):
        return path if path and os.path.exists(path) else None
    
    def _load_img_txt_model(self):
        backbone_weights = self._existing_path(
            os.environ.get(
                "LEXISG_DINOV3_BACKBONE_WEIGHTS",
                os.path.join(_CHECKPOINTS, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            )
        )
        text_weights = self._existing_path(
            os.environ.get(
                "LEXISG_DINOV3_TEXT_WEIGHTS",
                os.path.join(_CHECKPOINTS, "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"),
            )
        )

        load_kwargs = {}
        if backbone_weights is not None:
            load_kwargs["backbone_weights"] = backbone_weights
        if text_weights is not None:
            load_kwargs["weights"] = text_weights

        model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(**load_kwargs)
        return model.to(self.device), tokenizer


def _demo():
    parser = argparse.ArgumentParser(description="Smoke test for DINOv3 text-image loading.")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["office", "kitchen", "corridor"],
        help="Text prompts to encode after model load.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image path to encode with the text-image model.",
    )
    args = parser.parse_args()

    model = Dino()
    print(f"device={model.device}")
    print("dinov3 text-image model loaded")

    text_feats = model.get_txt_feat(args.prompts)
    print(f"text_prompts={args.prompts}")
    print(f"text_feat_shape={tuple(text_feats.shape)}")

    if args.image:
        image_feats = model.get_img_txt_feat([args.image])
        print(f"image={args.image}")
        print(f"image_feat_shape={tuple(image_feats.shape)}")
        sims = model.get_similarity(text_feats, image_feats)
        print(f"similarity_shape={tuple(sims.shape)}")
        print(f"top_prompt={args.prompts[int(np.argmax(sims[:, 0]))]}")


if __name__ == "__main__":
    _demo()
