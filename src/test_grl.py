import torch
from model_replay import ReplayResNet

def main():
    B = 4
    x = torch.randn(B,3,80,397)
    model = ReplayResNet(
        n_transform=2,
        n_domain=2,
        embedding_dim=256
    )
    model.train()
    (
        class_logits,
        transform_logits,
        domain_logits,
        embedding
    ) = model(x,grl_strength=1.0)

    print("Input:",x.shape)
    print("Class logits:",class_logits.shape)
    print("Transform logits:",transform_logits.shape)
    print("Domain logits:",domain_logits.shape)
    print("Embedding:",embedding.shape)

    class_labels = torch.tensor([0, 1, 0, 1])
    transform_labels = torch.tensor([0, 1, 1, 0])
    domain_labels = torch.tensor([-1, 0, 1, -1])
    cls_loss = torch.nn.functional.cross_entropy(class_logits,class_labels)
    transform_loss = (
        torch.nn.functional.cross_entropy(
            transform_logits,
            transform_labels
        )
    )

    mask = (domain_labels >= 0)
    domain_loss = (
        torch.nn.functional.cross_entropy(
            domain_logits[mask],
            domain_labels[mask]
        )
    )

    total_loss = (
        cls_loss
        + 0.10 * transform_loss
        + 0.05 * domain_loss
    )

    print()
    print("Loss")
    print("------------------------")
    print("Class loss    :",cls_loss.item())
    print("Transform loss:",transform_loss.item())
    print("Domain loss   :",domain_loss.item())
    print("Total loss    :",total_loss.item())

    total_loss.backward()
    print()
    print("Backward berhasil.")
    assert (class_logits.shape==(B, 2))
    assert (transform_logits.shape==(B, 2))
    assert (domain_logits.shape==(B, 2))
    assert (embedding.shape==(B, 256))

    print()
    print("====================================")
    print("TAHAP 7 - DOMAIN ADVERSARIAL")
    print("====================================")
    print("GRL dan masked domain loss berhasil.")

if __name__ == "__main__":
    main()