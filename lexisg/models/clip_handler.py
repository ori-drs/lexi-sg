import os

import torch 
import open_clip
from PIL import Image
import numpy as np

class CLIP: 
    def __init__(self): 
        print("Loading CLIP")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = self._load_model()
        self.model = self.model.eval().to(self.device)
        self.tokenizer = self._get_tokenizer()

    def _get_feat_dim(self): 
        return self.model.text_projection.shape[1]
    
    # def get_room_label(self, batch): 
    #     # for now choose the first image in the batch
    #     processed_image = self._preprocess_image(batch[2]) 
    #     image_feat = self._extract_image_feat(processed_image) 
    #     sim = self._get_sim(image_feat, self.room_text_feat) 
    #     room_label = self.room_prompt_list[np.argmax(sim)]
    #     return room_label

    @staticmethod
    def _is_image_batch(images):
        if isinstance(images, np.ndarray):
            return images.ndim == 4
        if isinstance(images, (list, tuple)):
            return len(images) > 0 and isinstance(images[0], (np.ndarray, Image.Image))
        return False

    @staticmethod
    def _to_pil_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.ndim != 3:
                raise ValueError(f"Expected image array with 3 dims, got shape={array.shape}")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array).convert("RGB")
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    def get_img_feat(self, images):
        if isinstance(images, (str, os.PathLike, Image.Image)):
            image_items = [images]
        elif isinstance(images, np.ndarray):
            image_items = list(images) if self._is_image_batch(images) else [images]
        elif isinstance(images, (list, tuple)):
            image_items = list(images)
        else:
            raise TypeError(f"Unsupported images input type: {type(images)!r}")

        image_tensors = torch.stack(
            [self.preprocess(self._to_pil_image(image)) for image in image_items], dim=0
        ).to(self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda")
            if self.device == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.no_grad(), autocast_ctx:
            img_features = self.model.encode_image(image_tensors)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        return img_features.detach().cpu().numpy()
    
    def get_txt_feat(self, txt): 
        tokenized_txt = self.tokenizer(txt).to(self.device)
        with torch.no_grad(), torch.autocast("cuda"):
            text_features = self.model.encode_text(tokenized_txt)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu().numpy()
    
    def get_txt_feat_batch(self, txt_list, batch_size=64):
        """
        Get the text features from the CLIP model
        :param in_text (list): the text to get the features from
        :param clip_model (CLIP): the CLIP model
        :param clip_feat_dim (int): the dimension of the CLIP features
        :param batch_size (int): the batch size for the inference
        :return: the text features
        """

        text_tokens = self.tokenizer(txt_list).to(self.device)
        text_id = 0
        text_feats = np.zeros((len(txt_list), self._get_feat_dim()), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(txt_list) - text_id, batch_size)
            text_batch = text_tokens[text_id : text_id + batch_size]
            with torch.inference_mode():
                batch_feats = self.model.encode_text(text_batch).float()
                batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
                batch_feats = np.float32(batch_feats.cpu())
                text_feats[text_id : text_id + batch_size, :] = batch_feats
                text_id += batch_size
        return text_feats
    
    def get_similarity(self, feat1, feat2):
        return feat1 @ feat2.T
    
    def _load_model(self): 
        return open_clip.create_model_and_transforms(
                        'ViT-H-14', 
                        pretrained='laion2b_s32b_b79k')
    
    def _get_tokenizer(self):
        return open_clip.get_tokenizer("ViT-H-14")

    def release(self):
        # Drop heavyweight references so GC can reclaim GPU memory promptly.
        if hasattr(self, "model"):
            try:
                del self.model
            except Exception:
                pass
        if hasattr(self, "tokenizer"):
            try:
                del self.tokenizer
            except Exception:
                pass
        if hasattr(self, "preprocess"):
            try:
                del self.preprocess
            except Exception:
                pass

    def _preprocess_image(self, image): 
        return self.preprocess(Image.open(image)).unsqueeze(0)

    # def _get_sim(self, image_feat, text_feat): 
    #     return (100.0 * image_feat @ text_feat.T).softmax(dim=-1).detach().numpy()
        
    # add entropy calc func
#TODO: refactor padding options
    def get_object_feat(self, tracker_id, masks, images, pad_ratio=None, pad_pixel=None, mode="original"):
        """
        For a given tracker_id, find the largest mask across frames, crop the RGB image
        to that mask (with padding), and extract a DINO feature.

        Args:
            tracker_id: The tracker ID whose masks we are interested in.
            masks: Dict[fid, Dict[tracker_id, mask ndarray]]
            images: Dict[fid, RGB image ndarray of shape (H, W, 3)]
            pad_ratio: Padding ratio to apply around the bounding box of the mask.
            pad_pixel: Fixed pixel padding to apply around the bounding box of the mask.
            mode: ["original", "black_bg", "white_bg"] How to handle background pixels.

        Returns:
            object_feat (np.ndarray) if successful, otherwise None.
            masked_rgb (np.ndarray): The cropped RGB image corresponding to the largest mask.
        """
        
        use_ratio = pad_ratio is not None and pad_ratio > 0
        use_pixel = pad_pixel is not None and pad_pixel >= 0
        
        assert use_ratio or use_pixel, "Either pad_ratio or pad_pixel must be specified and positive."
        assert not (use_ratio and use_pixel), "Only one of pad_ratio or pad_pixel should be specified."
        assert mode in ["original", "black_bg", "white_bg"], "Invalid mode for mask handling."
        
        # Collect all masks for this tracker_id across frames
        matching_masks = []
        for fid, mask_dict in masks.items():
            if tracker_id in mask_dict:
                matching_masks.append((fid, mask_dict[tracker_id]))

        if not matching_masks:
            return None

        # Find the mask with the largest area
        largest_mask_info = None
        largest_area = 0

        for fid, mask in matching_masks:
            mask_bool = mask.astype(bool)
            h, w, _ = images[fid].shape
            mask_area = mask_bool.sum()
            image_area = h * w

            # Skip unreasonably large masks if desired
            # if mask_area > 0.05 * image_area:
            #     continue

            if mask_area > largest_area:
                largest_area = mask_area
                largest_mask_info = (fid, mask_bool)

        # Skip if no valid mask found
        if largest_mask_info is None:
            return None

        fid, mask_bool = largest_mask_info
        rgb_image = images[fid]


        ys, xs = np.where(mask_bool)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        # Apply padding
        H, W, _ = rgb_image.shape
        if use_pixel:
            pad_y = pad_pixel
            pad_x = pad_pixel
        elif use_ratio:
            box_h = ymax - ymin
            box_w = xmax - xmin
            pad_y = int(box_h * (pad_ratio - 1) / 2)
            pad_x = int(box_w * (pad_ratio - 1) / 2)
        else: # shouldn't reach here
            pad_y = 0
            pad_x = 0
        ymin = max(0, ymin - pad_y)
        ymax = min(H, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(W, xmax + pad_x)

        # Mask and crop
        if mode == "white_bg":
            masked_rgb = np.ones_like(rgb_image) * 255
        elif mode == "black_bg":
            masked_rgb = np.zeros_like(rgb_image)
        else:
            masked_rgb = rgb_image.copy()
        
        masked_rgb[mask_bool, :] = rgb_image[mask_bool, :]
        cropped_rgb = masked_rgb[ymin:ymax, xmin:xmax]
        
        # Extract CLIP feature from the largest-mask crop
        feats = self.get_img_feat(cropped_rgb)
        return feats, cropped_rgb
