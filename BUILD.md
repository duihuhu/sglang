
```
cd python
pip install .
```

```
cd sgl-model-gateway

pip install maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"

make python-dev
```

```
pip install "sglang[afd-ucx]"
```

```
Mooncake缺失：
apt-get install -y libibverbs1 ibverbs-providers
```