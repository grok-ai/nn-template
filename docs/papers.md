---
hide:
  - navigation
  - toc
---

# Scientific Papers based on nn-template

??? hint "WikiNEuRal: Combined Neural and Knowledge-based Silver Data Creation for Multilingual NER."

    [![](https://shields.io/badge/-Repository-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/Babelscape/wikineural)

    ```bibtex
    @inproceedings{wikineural,
        title = "{W}iki{NE}u{R}al: {C}ombined Neural and Knowledge-based Silver Data Creation for Multilingual {NER}",
        author = "Tedeschi, Simone  and
          Maiorca, Valentino  and
          Campolungo, Niccol{\`o}  and
          Cecconi, Francesco  and
          Navigli, Roberto",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
        month = nov,
        year = "2021",
        address = "Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.findings-emnlp.215",
        pages = "2521--2533",
        abstract = "Multilingual Named Entity Recognition (NER) is a key intermediate task which is needed in many areas of NLP. In this paper, we address the well-known issue of data scarcity in NER, especially relevant when moving to a multilingual scenario, and go beyond current approaches to the creation of multilingual silver data for the task. We exploit the texts of Wikipedia and introduce a new methodology based on the effective combination of knowledge-based approaches and neural models, together with a novel domain adaptation technique, to produce high-quality training corpora for NER. We evaluate our datasets extensively on standard benchmarks for NER, yielding substantial improvements up to 6 span-based F1-score points over previous state-of-the-art systems for data creation.",
    }
    ```

??? hint "Named Entity Recognition for Entity Linking: What Works and What's Next."

    [![](https://shields.io/badge/-Repository-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/Babelscape/ner4el)

    ```bibtex
    @inproceedings{tedeschi-etal-2021-named-entity,
        title = "{N}amed {E}ntity {R}ecognition for {E}ntity {L}inking: {W}hat Works and What{'}s Next",
        author = "Tedeschi, Simone  and
          Conia, Simone  and
          Cecconi, Francesco  and
          Navigli, Roberto",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
        month = nov,
        year = "2021",
        address = "Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.findings-emnlp.220",
        pages = "2584--2596",
        abstract = "Entity Linking (EL) systems have achieved impressive results on standard benchmarks mainly thanks to the contextualized representations provided by recent pretrained language models. However, such systems still require massive amounts of data {--} millions of labeled examples {--} to perform at their best, with training times that often exceed several days, especially when limited computational resources are available. In this paper, we look at how Named Entity Recognition (NER) can be exploited to narrow the gap between EL systems trained on high and low amounts of labeled data. More specifically, we show how and to what extent an EL system can benefit from NER to enhance its entity representations, improve candidate selection, select more effective negative samples and enforce hard and soft constraints on its output entities. We release our software {--} code and model checkpoints {--} at https://github.com/Babelscape/ner4el.",
    }
    ```
