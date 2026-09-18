"""Command line interface for prediction and figure generation."""
import argparse
from .predict import process_file


def main():
    parser = argparse.ArgumentParser(prog="qdgt-nta")
    sub = parser.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict", help="Predict and optionally check applicability domain")
    pred.add_argument("--model", choices=("d", "pos", "neg"), required=True)
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--ad", action="store_true")
    pred.add_argument("--similarity", type=float)
    pred.add_argument("--count", type=int)
    figs = sub.add_parser("figures", help="Generate figures from the supplement")
    figs.add_argument("--workbook", default="supplementary file.xlsx")
    figs.add_argument("--output", default="figures")
    args = parser.parse_args()
    if args.command == "predict":
        process_file(args.input, args.output, args.model, ad=args.ad, similarity=args.similarity, count=args.count)
    else:
        from .figures import create_figures
        for name in create_figures(args.workbook, args.output):
            print(name)


if __name__ == "__main__":
    main()
