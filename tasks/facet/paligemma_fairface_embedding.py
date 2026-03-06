import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import pandas as pd
from tqdm import tqdm
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch.nn.functional as F

import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from config import DATA_DIR  # noqa: E402


def image_loader():
    fairface_dir = os.path.join(DATA_DIR, "fairface")
    train_csv = os.path.join(fairface_dir, "fairface_label_train.csv")
    val_csv = os.path.join(fairface_dir, "fairface_label_val.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"FairFace train CSV not found at {train_csv}. "
            "Please place the FairFace dataset under DATA_DIR/fairface "
            "(see README for download instructions)."
        )

    train_data = pd.read_csv(train_csv)
    valid_data = pd.read_csv(val_csv) if os.path.exists(val_csv) else None
    test_data = None

    gender_map = {"Female": 0, "Male": 1}
    race_map = {
        "East Asian": 0,
        "Indian": 1,
        "Black": 2,
        "White": 3,
        "Middle Eastern": 4,
        "Latino_Hispanic": 5,
        "Southeast Asian": 6,
    }
    age_map = {
        "0-2": 0,
        "3-9": 1,
        "10-19": 2,
        "20-29": 3,
        "30-39": 4,
        "40-49": 5,
        "50-59": 6,
        "60-69": 7,
        "more than 70": 8,
    }

    for df in [train_data, valid_data]:
        if df is None:
            continue
        df["gender"] = df["gender"].map(gender_map)
        df["race"] = df["race"].map(race_map)
        df["age"] = df["age"].map(age_map)

    return (
        train_data.reset_index(drop=True),
        valid_data.reset_index(drop=True) if valid_data is not None else None,
        test_data,
        fairface_dir,
    )


class ImageDataset(Dataset):
    def __init__(self, data, sens_idx, label_idx, root_dir=None, transform=None):
        self.transform = transform
        self.data = data
        self.root_dir = root_dir
        self.sens_idx = sens_idx
        self.label_idx = label_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name)
        if self.transform:
            prompt = ""
            image = self.transform(images=image, text=prompt, return_tensors="pt")
        else:
            image = [img_name]

        if isinstance(self.sens_idx, list):
            sens = self.data.loc[idx, self.sens_idx].values.astype(int)
        else:
            sens = self.data[self.sens_idx][idx]
            sens = max(int(sens), 0)
        if isinstance(self.label_idx, list):
            label = (self.data.loc[idx, self.label_idx].values > 0).astype(int)
        elif self.label_idx is None:
            label = None
        else:
            label = self.data[self.label_idx][idx]
            label = max(int(label), 0)
        if label is None:
            return image, sens
        elif sens is None:
            return image, label
        else:
            return image, sens, label


def main():
    print("Extract PaliGemma image and decoder embeddings on FairFace for FACET/ALA.")

    label = None
    sens = ["age", "gender", "race"]

    train_dataset, val_dataset, _, fairface_dir = image_loader()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_id = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = model.to(device)
    torch.cuda.empty_cache()
    model.eval()

    # Where to save embeddings
    embedding_dir = os.path.join(os.path.dirname(__file__), "embedding")
    os.makedirs(embedding_dir, exist_ok=True)

    # 1) Pooled vision-tower embeddings (optional; kept for parity with LLaVA script)
    save_path_train = os.path.join(embedding_dir, "fairface_paligemma_train.pt")
    if os.path.exists(save_path_train):
        print(f"[PaliGemma] Training vision embeddings already exist at: {save_path_train}. Skipping.")
    else:
        bs = 128
        train_data = ImageDataset(train_dataset, sens, label, fairface_dir, processor)
        train_loader = DataLoader(
            train_data,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

        with torch.no_grad():
            image_embeddings_list = []
            sensitive_attributes_list = []

            for x_batch, s_batch in tqdm(train_loader):
                x_batch = x_batch.to(device)
                image_outputs = model.vision_tower(
                    x_batch["pixel_values"].squeeze(1).to(torch.bfloat16),
                    output_hidden_states=True,
                )
                paligemma_image_embeds = image_outputs.last_hidden_state
                pooled_image_outputs = F.adaptive_avg_pool1d(
                    paligemma_image_embeds.permute(0, 2, 1),
                    1,
                ).squeeze(-1)

                image_embeddings_list.append(pooled_image_outputs.cpu())
                sensitive_attributes_list.append(s_batch)

        torch.save(
            {
                "image_embeddings": torch.cat(image_embeddings_list),
                "sensitive_attributes": torch.cat(sensitive_attributes_list),
            },
            save_path_train,
        )
        print(f"[PaliGemma] Training vision embeddings saved at: {save_path_train}")
        torch.cuda.empty_cache()

    # 2) Decoder embeddings (for SFID / CLIP-CLIP / DEAR)
    save_path_decoder = os.path.join(embedding_dir, "fairface_paligemma_train_decoder.pt")
    if os.path.exists(save_path_decoder):
        print(f"[PaliGemma] Training decoder embeddings already exist at: {save_path_decoder}. Skipping.")
    else:
        bs = 1
        train_data = ImageDataset(train_dataset, sens, label, fairface_dir, transform=False)
        train_loader = DataLoader(
            train_data,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )
        cnt = 0
        with torch.no_grad():
            decode_embeddings_list = []
            sensitive_attributes_list = []
            for x_batch, s_batch in tqdm(train_loader):
                prompt = "Describe the photo in detail"
                image = Image.open(x_batch[0][0]).convert("RGB")

                model_inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                ).to(torch.bfloat16).to(device)

                generation = model.generate(
                    **model_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                decode_hidden_states = torch.tensor(generation.hidden_states[0][0])
                decode_hidden_states = F.adaptive_avg_pool1d(
                    decode_hidden_states.permute(0, 2, 1),
                    1,
                ).squeeze(-1)

                decode_embeddings_list.append(decode_hidden_states.cpu())
                sensitive_attributes_list.append(s_batch)
                cnt += 1
                if cnt == 20000:
                    break

        torch.save(
            {
                "decode_embeddings": torch.cat(decode_embeddings_list),
                "sensitive_attributes": torch.cat(sensitive_attributes_list),
            },
            save_path_decoder,
        )
        print(f"[PaliGemma] Training decoder embeddings saved at: {save_path_decoder}")
        torch.cuda.empty_cache()

    # 3) Validation decoder embeddings (optional but useful for SFID/CLIP-CLIP tuning)
    if val_dataset is not None:
        save_path_val_decoder = os.path.join(embedding_dir, "fairface_paligemma_val_decoder.pt")
        if os.path.exists(save_path_val_decoder):
            print(f"[PaliGemma] Validation decoder embeddings already exist at: {save_path_val_decoder}. Skipping.")
        else:
            bs = 1
            val_data = ImageDataset(val_dataset, sens, label, fairface_dir, transform=False)
            val_loader = DataLoader(
                val_data,
                batch_size=bs,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
            cnt = 0
            with torch.no_grad():
                decode_embeddings_list = []
                sensitive_attributes_list = []
                for x_batch, s_batch in tqdm(val_loader):
                    prompt = "Describe the photo in detail"
                    image = Image.open(x_batch[0][0]).convert("RGB")

                    model_inputs = processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt",
                    ).to(torch.bfloat16).to(device)

                    generation = model.generate(
                        **model_inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
                    decode_hidden_states = torch.tensor(generation.hidden_states[0][0])
                    decode_hidden_states = F.adaptive_avg_pool1d(
                        decode_hidden_states.permute(0, 2, 1),
                        1,
                    ).squeeze(-1)

                    decode_embeddings_list.append(decode_hidden_states.cpu())
                    sensitive_attributes_list.append(s_batch)
                    cnt += 1
                    if cnt == 20000:
                        break

            torch.save(
                {
                    "decode_embeddings": torch.cat(decode_embeddings_list),
                    "sensitive_attributes": torch.cat(sensitive_attributes_list),
                },
                save_path_val_decoder,
            )
            print(f"[PaliGemma] Validation decoder embeddings saved at: {save_path_val_decoder}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

