import argparse
import os
import torch
import timm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--out_dir", type=str, default="./pretrained")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Info] Downloading pretrained model: {args.model}")
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=1000,
    )

    out_path = os.path.join(args.out_dir, f"{args.model}_imagenet_state_dict.pt")
    torch.save(model.state_dict(), out_path)

    print(f"[Info] Saved pretrained state_dict to:")
    print(out_path)


if __name__ == "__main__":
    main()