import os
import torch
import sys
sys.path.append("..")
from models.build_model import build_model

"""
Transfers model weights.
You can create a non-trained target model buy running:
python train.py --log_step 1 --max_step 1 ...
"""

trained_model_dir = "20220803-130921"
new_model_dir = "20220803-131016"

device = "cuda" if torch.cuda.is_available() else 'cpu'

main_dir = "../../output"

trained_config = torch.load(os.path.join(main_dir, trained_model_dir, "model_config.pt"))

trained_model, _ = build_model(None, load_config_dict=trained_config)
trained_model = trained_model.to(device)
trained_model.load_state_dict(torch.load(os.path.join(main_dir, trained_model_dir, 'model.pt'), map_location=device))

new_config = torch.load(os.path.join(main_dir, new_model_dir, "model_config.pt"))
new_model, _ = build_model(None, load_config_dict=new_config)
new_model = new_model.to(device)

trained_params = trained_model.named_parameters()
new_params = new_model.named_parameters()
dict_new_params = dict(new_params)
for name1, param1 in trained_params:
    if name1 in dict_new_params:

        if name1 == 'embedding.weight':
            # continuous_concat may have different sized embedding
            size1 = dict_new_params[name1].data.shape[1]
            size2 = param1.data.shape[1]
            size_transfer = min((size1, size2))
            dict_new_params[name1].data[:, :size_transfer] = param1.data[:, :size_transfer]
        else:
            dict_new_params[name1].data.copy_(param1.data)


output_path = os.path.join(main_dir, new_model_dir, 'model.pt')
torch.save(new_model.state_dict(), output_path)

print(f"Saved to {output_path}")
