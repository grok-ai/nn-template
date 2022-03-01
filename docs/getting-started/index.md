# Principles behind nn-template

When developing neural models ourselves, we often struggled with:

- **Reproducibility**. We strongly believe in the reproducibility requirement of scientific work.
- **Framework Learning**. Even when you find (or code yourself) the best framework to fit your needs, you still end up
  in messy situations when collaborating since others have to learn to use it;
- **Avoiding boilerplate**. We were bored to write the same code over and over in
    every project to handle the typical ML pipeline.

Over the course of the years, we fine-tuned our toolbox to reach this local minimum with respect to our self-imposed
requirements. After many epochs of training, the result is **nn-template**.


!!! warning "nn-template is not a framework"

    - It does not aim to sidestep the need to write code.
    - It does not contraint your workflow more than PyTorch Lightning does.

