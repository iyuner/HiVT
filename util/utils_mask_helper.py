import torch

def get_random_masked(data, config):
    if config.mask_strategy == "pointwise":
        return get_random_masked_pointwise(data, config.mask_ratio)
    # elif config.mask_strategy == "patchwise":
    #     return get_random_masked_patchwise(data, config.mask_ratio, config.patch_size)
    else:
        raise NotImplementedError

        
# data, original_padding_mask, addition_padding_mask = get_random_masked_pointwise(data, agents_out)
def get_random_masked_pointwise(data, mask_percentage):
    device = data['x'].device
    time_dim = 1
    
    
    gt_agents = torch.cat((data['x'], data['y']), dim=time_dim) # [N_flatten_whole_batch, 50, 2]
    mask = torch.rand((data['padding_mask'].shape)).to(device)
    mask = (mask > mask_percentage) # [N_flatten_whole_batch, 50]
    addition_padding_mask = ~mask   # True means, padded = invalid

    data['padding_mask'] = ~torch.logical_and(~data['padding_mask'], mask) # True means, padded = invalid
    
    data['bos_mask'] = torch.zeros(data['bos_mask'].shape[0], 50, dtype=torch.bool).to(device)
    data['bos_mask'][:, 0] = ~data['padding_mask'][:, 0]
    data['bos_mask'][:, 1: 50] = data['padding_mask'][:, : 49] & ~data['padding_mask'][:, 1: 50]

    data['x'] = gt_agents * (~data['padding_mask']).unsqueeze(-1)
    data['y'] = gt_agents

    return data, addition_padding_mask