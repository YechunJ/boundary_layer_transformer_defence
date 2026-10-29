import argparse
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--root", type=str, default="./data")
    args = parser.parse_args()

    if args.dataset == "cifar10":
        datasets.CIFAR10(root=args.root, train=True, download=True)
        datasets.CIFAR10(root=args.root, train=False, download=True)
    elif args.dataset == "cifar100":
        datasets.CIFAR100(root=args.root, train=True, download=True)
        datasets.CIFAR100(root=args.root, train=False, download=True)

    print(f"[Info] Downloaded {args.dataset} to {args.root}")


if __name__ == "__main__":
    main()