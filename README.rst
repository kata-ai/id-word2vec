Indonesian Word Vectors
=======================

This repository contains the code for our work:

Kurniawan, Kemal. 2019. “KaWAT: A Word Analogy Task Dataset for Indonesian.” ArXiv:1906.09912 [Cs], June. http://arxiv.org/abs/1906.09912.

Requirements
------------

Create a virtual environment from ``environment.yml`` file using conda::

    conda env create -f environment.yml

This will create a virtual environment named ``id-word2vec``. Next, activate the virtual environment. To train with GloVe, the GloVe binaries must be installed as well. Refer to the `GloVe repository <https://github.com/stanfordnlp/GloVe/>`_ for instructions.

.. note:: All the scripts use `Sacred <https://github.com/IDSIA/sacred>`_ so they can be run with ``help`` and ``print_config`` command to show help and list of configurations respectively.

Evaluate embeddings against analogies
-------------------------------------

Simply run::

    ./run_evaluation.py with vectors_path=vectors.txt analogy_path=analogy.txt

This command computes 95% bootstrap CI of accuracy at rank 1 of solving the analogy task in ``analogy.txt`` with word vectors in ``vectors.txt``. The analogy file must be formatted like Google Word Analogy: each line contains 4 words separated by whitespaces corresponding to ``A : B :: C : D`` analogy.

Setting up Mongodb observer
---------------------------

Sacred allows the experiments to be observed and saved to a Mongodb database. The experiment scripts above can readily be used for this, simply set two environment variables ``SACRED_MONGO_URL`` and ``SACRED_DB_NAME`` to your Mongodb authentication string and database name (to save the experiments into) respectively. Once set, the experiments will be saved to the database. Use ``-u`` flag when invoking the experiment script to disable saving.

Citation
--------

If you're using our work, please cite::

     @article{kurniawan2019,
       title={KaWAT: A Word Analogy Task Dataset for Indonesian},
       url={http://arxiv.org/abs/1906.09912},
       journal={arXiv:1906.09912 [cs]},
       author={Kurniawan, Kemal},
       year={2019},
       month={Jun}
     }
