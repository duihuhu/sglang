import torch

from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_lf_pf,
    transfer_kv_per_layer_pf_lf,
)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.float32
    layer_num = 2
    device_size = 32
    host_size = 16
    head_num = 1
    head_dim = 2
    item_size = head_num * head_dim * torch.tensor([], dtype=dtype).element_size()
    layout_dim = item_size * layer_num

    dev_k_layers = []
    dev_v_layers = []
    for layer_id in range(layer_num):
        k = torch.zeros((device_size, head_num, head_dim), dtype=dtype, device=device)
        v = torch.zeros_like(k)
        for idx in range(device_size):
            k[idx, 0, :] = 1000 * (layer_id + 1) + idx * 10 + torch.arange(
                head_dim, device=device
            )
            v[idx, 0, :] = 2000 * (layer_id + 1) + idx * 10 + torch.arange(
                head_dim, device=device
            )
        dev_k_layers.append(k)
        dev_v_layers.append(v)

    dev_k_ptrs = torch.tensor(
        [x.data_ptr() for x in dev_k_layers], dtype=torch.uint64, device=device
    )
    dev_v_ptrs = torch.tensor(
        [x.data_ptr() for x in dev_v_layers], dtype=torch.uint64, device=device
    )

    host_k = torch.empty(
        (host_size, layer_num, head_num, head_dim), dtype=dtype, pin_memory=True
    )
    host_v = torch.empty_like(host_k, pin_memory=True)
    host_k.zero_()
    host_v.zero_()

    device_indices = torch.tensor([20, 22, 25, 21, 23, 24], dtype=torch.int64, device=device)
    host_indices = torch.tensor([2, 3, 4, 8, 9, 10], dtype=torch.int64, device=device)

    transfer_kv_all_layer_lf_pf(
        src_k_layers=dev_k_ptrs,
        dst_k=host_k,
        src_v_layers=dev_v_ptrs,
        dst_v=host_v,
        src_indices=device_indices,
        dst_indices=host_indices,
        item_size=item_size,
        dst_layout_dim=layout_dim,
        num_layers=layer_num,
    )
    torch.cuda.synchronize()

    for pos, dev_idx in zip(host_indices.cpu().tolist(), device_indices.cpu().tolist()):
        for layer_id in range(layer_num):
            assert torch.equal(
                host_k[pos, layer_id, 0, :].cuda(),
                dev_k_layers[layer_id][dev_idx, 0, :],
            )
            assert torch.equal(
                host_v[pos, layer_id, 0, :].cuda(),
                dev_v_layers[layer_id][dev_idx, 0, :],
            )

    restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
    restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]
    for layer_id in range(layer_num):
        transfer_kv_per_layer_pf_lf(
            src_k=host_k,
            dst_k=restored_k_layers[layer_id],
            src_v=host_v,
            dst_v=restored_v_layers[layer_id],
            src_indices=host_indices,
            dst_indices=device_indices,
            layer_id=layer_id,
            item_size=item_size,
            src_layout_dim=layout_dim,
        )
    torch.cuda.synchronize()

    for layer_id in range(layer_num):
        assert torch.equal(
            restored_k_layers[layer_id][device_indices],
            dev_k_layers[layer_id][device_indices],
        )
        assert torch.equal(
            restored_v_layers[layer_id][device_indices],
            dev_v_layers[layer_id][device_indices],
        )

    print("real kvcacheio single-extent page_first roundtrip OK")


if __name__ == "__main__":
    main()
