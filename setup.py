import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dx-llava",
    version="1.0.0",
    author="Tuan Nguyen",
    author_email="",
    description="DX-LLaVA: Large Language and Vision Assistant with ConvNeXt for Deepfake Detection and eXplanation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.37.0",
        "tokenizers>=0.15.0",
        "sentencepiece",
        "shortuuid",
        "httpx==0.24.0",
        "deepspeed>=0.12.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
        "pydantic",
        "markdown2[all]",
        "numpy",
        "requests",
        "Pillow",
        "wandb",
        "gradio",
        "accelerate>=0.21.0",
        "einops",
        "flash-attn>=2.3.0",
    ],
)