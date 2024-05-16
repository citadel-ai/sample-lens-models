Download the pretrained model checkpoint from the link below to this folder:
https://drive.google.com/file/d/1kR5QuRAuooYcTNLMnMj80Z9IgSs8jtLO/view?usp=sharing

The model checkpoint file name should be `UNETR_model_best_acc.pth`.

Run the following command to build the Docker image:
```bash
docker build --platform linux/amd64 -t unetr-btcv-sample:latest .
```

Save the image to a `tar` file that can be uploaded to Lens:
```bash
docker save unetr-btcv-sample:latest -o <output_folder>/unetr-btcv-sample.amd64.tar
```
