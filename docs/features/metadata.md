# MetaData

The *bridge* between the Lightning DataModule and the Lightning Module.

It is responsible for collecting data information to be fed to the module.
The Lightning Module will receive an instance of MetaData when instantiated,
both in the train loop or when restored from a checkpoint.

!!! warning

    MetaData exposes `save` and `load`. Those are two user-defined methods that specify how to serialize and de-serialize the information contained in its attributes.
    This is needed for the checkpointing restore to work properly and **must be
    always implemented**, where the metadata is needed.

This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
DataModule/Trainer independent (useful in prediction scenarios).
Examples are the class names in a classification task or the vocabulary in NLP tasks.

