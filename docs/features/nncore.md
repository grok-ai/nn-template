# NN Template core

Most of the logic is abstracted from the template into an accompanying library: [`nn-tempalte-core`](https://pypi.org/project/nn-template-core/).

This library contains the logic necessary for the restore, logging, and many other functionalities implemented in the template.

!!! info

    This decoupling eases the updating of the template, reaching a desiderable tradeoff:

    - `template`: easy to use and customize, hard to update
    - `library`: hard to customize, easy to update

    With our approach updating most of the functions is extremely easy, it is just a python
    dependency, while maintaing the flexibility of a template.


!!! warning

    It is important to not remove the `NNTemplateCore` callback from the instantiated callbacks
    in the template. It is used to inject personalized behaviour in the training loop.
