"""Convenience entry point for retraining the shipped model."""
from income_classifier.cli import main

if __name__ == "__main__":
    main(["train", "--classifier", "hgb", "--iterations", "50"])
