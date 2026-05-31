# oil-gas-predictive-maintenance

## Run locally

Start the local web server:

```bash
python local_server.py
```

Then open:

```bash
http://127.0.0.1:8000
```

This serves both the frontend and the `/api/predict` endpoint on the same origin.

## Deploy on Vercel

Use the `Other` preset. Vercel installs production dependencies from `uv.lock`; the project uses `xgboost-cpu` to avoid the large CUDA/NCCL packages pulled by the default `xgboost` wheel.
