from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from TextSummerizer.entity import ModelTrainerConfig
import torch
import os


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step} - Loss: {logs.get('loss')}, Learning Rate: {logs.get('learning_rate')}")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)
 
        trainer_args = TrainingArguments(
             output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
             per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
             weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
             eval_steps=self.config.eval_steps, save_steps=self.config.save_steps,
             gradient_accumulation_steps=self.config.gradient_accumulation_steps
         ) 


        ''' trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
         ) '''

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()


                ## ✅ Save the model and tokenizer properly
        model_dir = self.config.model_path
        tokenizer_dir = self.config.tokenizer_path

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)

        model_pegasus.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)

