import os

DATA_DIR = "data"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def download_imagenette(data_dir: str, url: str):
    if os.path.exists(data_dir):
        raise FileExistsError(
            f"{data_dir}/ already exists, delete it if you want to override it"
        )

    os.system(f"curl {url} -o imagenette.tgz")
    os.system("tar zxvf imagenette.tgz")
    os.system(f"mv imagenette2-160 {data_dir}")
    os.system("rm -fr imagenette.tgz")
    os.system(f"rm -f {os.path.join(data_dir, 'noisy_imagenette.csv')}")


if __name__ == "__main__":
    download_imagenette(DATA_DIR, IMAGENETTE_URL)
