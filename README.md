# TCI (Triton Captcha Inference)

Repository with artifacts for create docker image for inference TrOCR model with usage Triton Server

## Repository structure:

- **src/**: base scripts for inference
  - **model.py**: script with triton inference model;
  - **utils_data.py**: scripts for preprocessing and postprocessing datas;
  - **config.pbtxt**: config for configuring triton inference server;
  - **client.py**: example of client for usage triton inference server.

