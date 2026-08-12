import argparse

from rag.chain import ask
from rag.ingest import build_index


def main():
    parser = argparse.ArgumentParser(description="Local RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Index documents from the data directory")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str)

    args = parser.parse_args()

    if args.command == "ingest":
        build_index()
    elif args.command == "query":
        print(ask(args.question))


if __name__ == "__main__":
    main()
