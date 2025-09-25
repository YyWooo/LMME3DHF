from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_MLP
from safetensors.torch import load_file
import os
import shutil

model_path = ""
model = Qwen2_5_VL_MLP.from_pretrained(
                model_path,
                torch_dtype="bfloat16",
                device_map="auto"
    )

folder_name = os.path.split(model_path)[0]
extra_layers = load_file(os.path.join(folder_name, "mlp_layers.safetensors"))
missing_keys, unexpected_keys = model.load_state_dict(extra_layers, strict=False)
assert len(unexpected_keys) == 0, \
    f"Missing keys: {unexpected_keys}"


save_path = os.path.join(folder_name, "final_model")
model.save_pretrained(save_path)

for file_name in os.listdir(model_path):
    original_file_path = os.path.join(model_path, file_name)
    new_file_path = os.path.join(save_path, file_name)

    # Copy the file if it doesn't exist in the new folder
    if not os.path.exists(new_file_path):
        shutil.copy(original_file_path, new_file_path)
        print(f"Copied: {file_name}")
    else:
        print(f"Skipped (already exists): {file_name}")
