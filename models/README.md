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

The optimization process for this is run through `./scripts/main.sh -a min_gen`. Right now, the hyperparameters are fixed in that script, but they should probably be changed to arguments to the script.
The optimization is best run on a GPU.  

## Config Files 
All configs are .jsonnet files, which contain fields that `allennlp` reads from to instatiate a model. They make experimentation easy and reproducible. 
Generally, each field will contain a `type: name` entry, which directs allennlp to instantiate a subclass for that field based on the name.  
Each file corresponds roughly to one experiment.
At the top of the file are `local` variables that are defined for the whole file and re-used frequently. 
Each config has parameters for the following fields:
1. `dataset_reader` tells allennlp which dataset reader to use. The `type: vqav2` tells allennlp to instatiate a `VQAv2Reader`, since that's the registered name of that class: 

    ```
    @DatasetReader.register("vqav2")
    class VQAv2Reader(VisionReader):
    ```

The `validation_data_reader` is basically the same, but points to the validation path. 
These dataset readers take care of ingesting the VQA data, tokenizing it, batching and padding it together, and transferring it to the correct device (e.g. CUDA, CPU).
When multiple GPUs are available, they handle transferring it to the correct CUDA device.  
Some important params are `max_instances`, which is useful for debugging (setting it to $n$ means the reader will stop after $n$ instances). 
2. `model` constructs the model from named modules. The modules are all registered subclasses, where components are defined in the signature for `__init__()` function for the parent module.
For example `RSAVQAModel` has a `VisionLanguageEncoder` as a parameter, which is matched to the `vision_language_encoder` param in the config file.  
This is where we make modifications to the model architecture (number of layers, dropout, layer sizes, etc. )

3. `data_loader` is the module that handles the batch size, whether to shuffle elements before batching, etc. 
Not much to modify here. 

4. `distributed` is present if the number of gpus is more than 1. It handles distributed training (data parallel) across gpus. 
5. `trainer` is the trainer class that is used to train the model. 
This class contains the training loop (model.forward, loss.backward, optimizer.step) and defines which optimizer and learn-rate schedule to use. 
Hyperparameters like LR, schedule, number of steps, etc. are defined here. 


## Key files 
- `models/allennlp/data/dataset_readers/vqav2.py`: the dataset reader for the VQA data
- `models/allennlp/data/dataset_readers/vision_reader.py`: the parent class for the VQA dataset reader.
- `models/allennlp/models/rsa_vqa.py`: the main model class. The model has 3 main components: a vision-language encoder, a speaker module, and a listener module.
    - `models/allennlp/modules/vision/vision_language_encoder.py`: contains the subclasses for different vision-language encoders
    - `models/allennlp/modules/rsa_vqa/speaker.py`: contains the "speaker" module, which models the speaker in the RSA framework. Implementationally, this is an encoder-decoder, where the encoder projects the meaning vector and the decoder produces an utterance based on it. 
    - `models/allennlp/modules/rsa_vqa/listener.py`: acts as the "listener" in the RSA framework, takes in the output of the encoder from the vision-language encoder (or the previous speaker) and passes it through an MLP. 
- `models/allennlp/training/trainer.py`: defines the trainer class. Usually, not a lot of modifications to make here, but good for debugging. 
- `models/allennlp/predictors/rsa_vqa.py`: predictor class for the model, which predicts outputs at test time for an image and question input. 
- `models/allennlp/commands/min_gen.py`: contains the command for doing the gradient-based minimization on the meaning vector and then the subsequent generation through the speaker module. Relies on the some utilities:
    - `models/allennlp/util.py`: the function `minimize_and_generate()` defines the procedure for doing the mimimization and then generation. 