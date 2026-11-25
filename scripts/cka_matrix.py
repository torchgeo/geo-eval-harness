"""Write the linear CKA matrix between classification embeddings saved with output.embeddings."""

import argparse
import logging
from pathlib import Path

from geo_eval_harness.embeddings import MAX_CKA_SAMPLES, cka_matrix

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Compare the given embedding files and write their CKA matrix as CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Embedding files to compare.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=MAX_CKA_SAMPLES)
    parser.add_argument("--output", type=Path, required=True, help="CSV file to write.")
    args = parser.parse_args(argv)
    cka_matrix(args.paths, split=args.split, max_samples=args.max_samples).to_csv(args.output)
    logger.info("Wrote the CKA matrix of %d files to %s", len(args.paths), args.output)


if __name__ == "__main__":
    main()
