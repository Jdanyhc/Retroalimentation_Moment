import evaluate
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from datasets import load_dataset


model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset = load_dataset("wikiann", "es")
label_names = dataset["train"].features["ner_tags"].feature.names
tokens = tokenizer(dataset["train"][0]["tokens"], is_split_into_words=True)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

labels = dataset["train"][0]["ner_tags"]
word_ids = tokens.word_ids()

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = collator([tokenized_datasets["train"][tokens] for tokens in range(2)])

metric = evaluate.load("seqeval")

labels = dataset["train"][0]["ner_tags"]
labels = [label_names[event] for event in labels]

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


model_2 = AutoModelForTokenClassification.from_pretrained(model_checkpoint,id2label={i: label for i,label in enumerate(label_names)},label2id={v: k for k, v in id2label.items()})

model_2.config.num_labels
notebook_login()


args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    report_to="wandb"
)

small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
small_test = tokenized_datasets["test"].shuffle(seed=42).select(range(50))

trainer = Trainer(
    model=model_2,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()




