"""Train the model."""
from copy import deepcopy

import torch

from config import model_dir, base_model_name
from dataset import load
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM, AutoModelForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling

# BEGIN CONFIG #
# Hyperparameters
OPTIMISER = 'adamw_8bit'
SCHEDULER = 'cosine'
MAX_STEPS = 1_000_000
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 48_000
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98
ADAM_EPSILON = 1e-6
MAX_GRAD_NORM = 1.0
MLM_PROBABILITY = 0.15
FP16 = True

# Flags
RESUME = False
# END CONFIG #

if __name__ == '__main__':
    # Load the tokeniser.
    tokeniser: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(model_dir)
    
    # Load the base model.
    model: RobertaForMaskedLM = AutoModelForMaskedLM.from_pretrained(
        base_model_name,
        bos_token_id = tokeniser.bos_token_id,
        eos_token_id = tokeniser.eos_token_id,
        pad_token_id = tokeniser.pad_token_id,
    )

    # Identify the indices of tokens in the new model's vocabulary that are also present in the base model's vocabulary.
    base_model_vocabulary: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(base_model_name).get_vocab()
    vocabulary = tokeniser.get_vocab()
    shared_token_ids = {vocabulary[token] for token in base_model_vocabulary.keys() & vocabulary.keys()}

    # Calculate the average of every weight in the base model's token embedding layer.
    average_embedding = model.roberta.embeddings.word_embeddings.weight.mean(0)

    # Preserve the base model's token embedding layer.
    base_model_wte = deepcopy(model.roberta.embeddings.word_embeddings.weight.data)

    with torch.inference_mode():
        for i in range(len(model.roberta.embeddings.word_embeddings.weight)):
            # Overwrite the weights of tokens that are not shared with the base model with their averages in the base model's token embedding layers.
            if i not in shared_token_ids:
                new_weight = average_embedding

            # Reuse the weights of tokens that are shared with the base model's vocabulary.
            else:
                new_weight = base_model_wte[base_model_vocabulary[tokeniser.convert_ids_to_tokens(i)]]

            model.roberta.embeddings.word_embeddings.weight[i].copy_(new_weight)
            model.lm_head.decoder.weight[i].copy_(new_weight)
    
    # Convert the model into a Better Transformer in order to accelerate training.
    original_model_save_pretrained = model.save_pretrained
    model = model.to_bettertransformer()
    
    def better_transformer_save_pretrained(*args, **kwargs):
        """Overrides a model's `save_pretrained` method in order to convert it into a vanilla Transformer before saving it."""
        global model
        
        # Convert the model back into a vanilla Transformer if it isn't one already.
        if hasattr(model, 'use_bettertransformer') and model.use_bettertransformer is True:
            model = model.reverse_bettertransformer()
        
        # Save the model.
        output = original_model_save_pretrained(*args, **kwargs)
        
        # Convert the model back into a Better Transformer.
        model = model.to_bettertransformer()
        
        return output

    model.save_pretrained = better_transformer_save_pretrained

    # Load the data.
    data = load()

    # Initialise the trainer.
    training_args = TrainingArguments(
        optim = OPTIMISER,
        lr_scheduler_type = SCHEDULER,
        max_steps = MAX_STEPS,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        warmup_steps = WARMUP_STEPS,
        adam_beta1 = ADAM_BETA1,
        adam_beta2 = ADAM_BETA2,
        adam_epsilon = ADAM_EPSILON,
        per_device_train_batch_size = BATCH_SIZE,
        max_grad_norm = MAX_GRAD_NORM,
        fp16 = FP16,
        
        evaluation_strategy = 'steps',
        eval_steps = 16_384,
        per_device_eval_batch_size = BATCH_SIZE,

        save_strategy = 'steps',
        save_steps = 16_384,
        save_total_limit = 5,

        logging_strategy = 'steps',
        logging_steps = 128,
        logging_first_step = True,
        report_to = 'wandb',

        dataloader_pin_memory = True,
        output_dir = model_dir,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = data['train'],
        eval_dataset = data['validation'],
        data_collator = DataCollatorForLanguageModeling(tokeniser, mlm = True, mlm_probability = MLM_PROBABILITY),
    )
    
    if FP16:
        original_trainer_evaluate = trainer.evaluate
        
        def fp16_evaluate(*args, **kwargs):
            with torch.cuda.amp.autocast():
                return original_trainer_evaluate(*args, **kwargs)
        
        trainer.evaluate = fp16_evaluate
    
    # Train the model.
    trainer.train(
        resume_from_checkpoint = RESUME,
    )
    
    # Save the model.
    trainer.save_model()
    trainer.save_state()

    # Evaluate the model on the test sets.
    trainer.evaluate(data['test'])