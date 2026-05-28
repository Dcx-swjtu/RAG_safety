from setuptools import find_packages, setup


setup(
    name="verirag",
    version="1.0.0",
    description="Verification-guided RAG defense with RL policy selection.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "retrieval": ["sentence-transformers>=2.2.0", "faiss-cpu>=1.7.4"],
        "qwen": ["vllm>=0.2.0", "accelerate>=0.24.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
