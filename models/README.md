## Models

### Organization
- ./allennlp
    - is a clone of allennlp but from [this](https://github.com/allenai/allennlp/tree/7aa7e0e9537cc1e3f854e11b276c6d7063d02edc/allennlp/data/dataset_readers) commit, which has the VQA dataset reader (the most current version of master does not)
    - if they ever pull that commit into master, then we can do a diff on this directory and just import allennlp instead of copying it 
- ./allennlp_models
    - is a clone of allenlp_models from [this](https://github.com/allenai/allennlp-models/tree/1e89d5e51cb45f3e77a48d4983bf980088334fac/allennlp_models/generation/modules/seq_decoders) commit 
    - same story as above 
- ./config
    - contains the config jsonnet files needed for training 
- ./scripts 
    - scripts for running models, preprocessing data 
    - ./scripts/main.sh is the script for training and evaluation 
- ./slurm_scripts
    - scripts for running models on the grid 
- ./test_fixtures
    - contains some small subsets of data for testing purposes 

### Training a model 
Model training is done via `scripts/main.sh`. The command to train is `./scripts/main.sh -a train`. Training requires two environment variables: `CHECKPOINT_DIR` (where the model checkpoint will be written) and `TRAINING_CONFIG` (the .jsonnet config file for training). 

When training on the grid, a model can be trained by setting those environment variables and then running `sbatch slurm_scripts/train.sh --export`. 
**NB** you cannot train a model with a `CHECKPOINT_DIR` that already exists. 

### Evaluating a model 
The command for model evaluation is `./scripts/main.sh -a eval`. An additional environment variable is needed here: `TEST_DATA`, which is set to the name/path of the data being tested on. 

### Generating a minimum meaning vector 
The main idea in the project is to look at alternative questions for a given image and answer. The first way we're trying to do this is to freeze a trained model and do a gradient-based minimization of the image-question encoding (the "meaning vector") with respect to the VQA loss.
In other words, we want to find the meaning vector that would have given us the answer that we got. 
In some cases, that meaning vector will be the same after optimization as before. 
For example, if there's a picture of a yellow flower and the question is "what color is the flower" and the answer is "yellow", we would expect the minimum meaning vector to correspond to the same question as the original. 

However, in the dataset, we also have examples like this:

*Question*: "Was a beverage served?"
![Picture of a coffee cup](https://cs.jhu.edu/~esteng/images_for_hit/COCO_train2014_000000460694.jpg)

*Answers*: "yes", "yes", "coffee", "latte" 

In the case of "coffee" and "latte", the annotators are answering a slightly different question than the people who said "yes". 
Here, the minimum meaning vector would hopefully result in an output like "what beverage was served?" 








