Download the pretrained model checkpoint [from Google Drive](https://drive.google.com/file/d/1tcXvzZdLja2ni7PjMpJgfLVryZ0J5WtR/view?usp=sharing), and save it in this folder.

The model checkpoint file name should be `UNETR_model_best_acc.pth`.

Run the following command to build the Docker image for testing locally:
```bash
docker build -t unetr-btcv-sample:latest .
```

Run the following command to build the Docker image for Lens:
```bash
docker build --platform linux/amd64 -t unetr-btcv-sample-amd64:latest .
```

Save the image to a `tar` file that can be uploaded to Lens:
```bash
docker save unetr-btcv-sample-amd64:latest -o <output_folder>/unetr-btcv-sample.amd64.tar
```
