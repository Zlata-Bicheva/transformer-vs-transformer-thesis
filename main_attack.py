
# 1. Debug / early prints
print("STARTING SCRIPT")

# 2. Special libraries that may affect runtime (SBERT, torch)
from sentence_transformers import SentenceTransformer, util
import torch

# 3. Standard library
import os

# 4. Third-party general libs
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

# 5. Deep learning / transformers
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForMaskedLM, 
    logging
)

logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)


print("creating model")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("created model")


LABEL_TRUTHFUL = 0
LABEL_DECEPTIVE = 1
SIM_THRESHOLD = 0.60

# LIWC-inspired attack:
ATTACK_PROMPT_TEMPLATE = (
    "Rewrite the following deceptive hotel review such that it's classified"
    "as truthful by LLM-based deception classifiers."

    f"Make sure to keep semantic similarity above threshold - {SIM_THRESHOLD}"

    "Adjust only the writing style so that it aligns with linguistic patterns " 
    "associated with truthful reviews, as operationalized through " 
    "psycholinguistic features:"
        "- perceptual process language is slightly increased"
        "- affective intensity is reduced"
        "- self-referential pronouns are used naturally"
        "- certainty expressions are balanced"

    "Hotel review: {review}"
)

class GPT2Attacker:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_candidate(self, review_text):
        prompt = ATTACK_PROMPT_TEMPLATE.format(review=review_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # remove prompt part
        candidate = text[len(prompt):].strip()

        return candidate, None

class BERTAttacker:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_candidate(self, text):
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) < 5:
            return text, None

        idx = random.randint(1, len(tokens) - 2)
        tokens[idx] = self.tokenizer.mask_token

        masked_text = self.tokenizer.convert_tokens_to_string(tokens)
        
        inputs = self.tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        
        # filter inputs
        model_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        
        mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=False)

        if mask_positions.size(0) == 0:
            return text, None

        mask_pos = mask_positions[0, 1]
        logits = outputs.logits[0, mask_pos]

        probs = torch.softmax(logits, dim=-1)
        
        replacement_id = torch.topk(probs, k=5).indices[random.randint(0, 4)].item()
        replacement_token = self.tokenizer.decode([replacement_id]).strip()

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids[idx] = replacement_id
        candidate = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        return candidate, None

class LLMAttacker:
    def __init__(self, generate_fn):
        self.generate_fn = generate_fn

    def generate_candidate(self, review_text):
        prompt = ATTACK_PROMPT_TEMPLATE.format(review=review_text)
        return self.generate_fn(prompt)


class EncoderOnlyClassifier:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    # @torch.no_grad()
    def predict(self, text):
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            logits = self.model(**enc).logits.squeeze(0)
            pred = logits.argmax().item()

        return pred, logits

class GPT2Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt2.config.n_embd, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]  # [B, T, H]

        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0).long()  # safe

        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        cls_repr = hidden[batch_idx, lengths]  # last real token

        logits = self.classifier(self.dropout(cls_repr))
        return logits


class DecoderOnlyClassifier:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    # @torch.no_grad()
    def predict(self, text):
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            ).to(self.device)

            logits = self.model(enc["input_ids"], enc["attention_mask"]).squeeze(0)
            pred = logits.argmax().item()

        return pred, logits

class EncoderDecoderClassifier:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text):
        with torch.no_grad():
            prompt = f"Classify the statement as truthful or deceptive: {text}"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            #  Create decoder input (start token)
            decoder_input_ids = torch.tensor(
                [[self.model.config.decoder_start_token_id]],
                device=self.device
            )

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids
            )

            # logits for first generated token
            first_token_logits = outputs.logits[0, 0, :]

            # Map to binary labels
            truthful_ids = self.tokenizer("truthful").input_ids
            deceptive_ids = self.tokenizer("deceptive").input_ids

            truthful_score = first_token_logits[truthful_ids[0]]
            deceptive_score = first_token_logits[deceptive_ids[0]]

            binary_logits = torch.stack([truthful_score, deceptive_score])

            pred = binary_logits.argmax().item()

        return pred, binary_logits

def run_attack(
    original_text,
    attacker,
    victim_model,
    sbert_model,
    optimizer,
    target_label=0,        # truthful
    sim_threshold=SIM_THRESHOLD,
    query_budget=10,
):
    history = []
    current_text = original_text

    orig_emb = sbert_model.encode(original_text, convert_to_tensor=True)

    for step in range(query_budget):
        candidate, logprob = attacker.generate_candidate(current_text)

        # Reject empty candidates
        if candidate is None or candidate.strip() == "":
            history.append({
                "step": step,
                "text": candidate,
                "prediction": None,
                "confidence": None,
                "similarity": None,
                "accepted": False,
                "reason": "empty_candidate"
            })

            continue

        # SBERT similarity check (original anchoring)
        cand_emb = sbert_model.encode(candidate, convert_to_tensor=True)
        sim = util.cos_sim(orig_emb, cand_emb).item()

        if sim < sim_threshold:

            history.append({
                "step": step,
                "text": candidate,
                "prediction": None,
                "confidence": None,
                "similarity": sim,
                "accepted": False,
                "reason": "below_similarity_threshold"
            })
            continue

        pred, logits = victim_model.predict(candidate)
        probs = torch.softmax(logits, dim=-1)

        conf_reward = probs[target_label] - probs[1 - target_label]

        # Optional RL reward for valid candidate
        if logprob is not None:
            # reward = probs[target_label].detach() - 0.5
            loss = -conf_reward * logprob
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history.append({
            "step": step,
            "text": candidate,
            "prediction": pred,
            "confidence": probs[target_label].item(),
            "similarity": sim
        })

        if pred == target_label and sim >= sim_threshold:
            # pbar.close()
            return {
                "success": True,
                "steps": step + 1,
                "adversarial_text": candidate,
                "similarity": sim,
                "history": history
            }

        current_text = candidate

    # pbar.close()
    return {
        "success": False,
        "steps": query_budget,
        "similarity": None,
        "history": history
    }

def main():
    print("Loading victim models...")

    # Victims
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        "./saved_models/encoder_only_roberta"
    )
    encoder_only = EncoderOnlyClassifier(roberta_model, roberta_tokenizer, device)

    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    gpt2_model = GPT2Discriminator()
    state_path = "./saved_models/decoder_only/pytorch_model.bin"
    gpt2_model.load_state_dict(torch.load(state_path, map_location=device))
    gpt2_model.to(device).eval()

    decoder_only = DecoderOnlyClassifier(gpt2_model, gpt2_tokenizer, device)

    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "./saved_models/flan_t5_deception"
    )
    encoder_decoder = EncoderDecoderClassifier(t5_model, t5_tokenizer, device)

    victim_models = {
        "encoder_only": encoder_only,
        "decoder_only": decoder_only,
        "encoder_decoder": encoder_decoder
    }

    # Attackers
    print("Initializing attackers...")

    ATTACKER_ID = "google/flan-t5-small"
    attacker_tokenizer = AutoTokenizer.from_pretrained(ATTACKER_ID)
    attacker_model = AutoModelForSeq2SeqLM.from_pretrained(ATTACKER_ID).to(device)
    optimizer = torch.optim.Adam(attacker_model.parameters(), lr=1e-5)
    attacker_model.train()

    def generate_with_logprob(prompt):
        # inputs = attacker_tokenizer(prompt, return_tensors="pt").to(device)
        
        inputs = attacker_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            gen_out = attacker_model.generate(**inputs, max_new_tokens=200)

        gen_tokens = gen_out[:, -200:]
        text = attacker_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        labels = gen_tokens.clone()
        labels[labels == attacker_tokenizer.pad_token_id] = -100

        out = attacker_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        seq_logprob = -out.loss * (labels != -100).sum()
        return text, seq_logprob

    t5_attacker = LLMAttacker(generate_with_logprob)


    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id  #  FIX

    gpt_attacker = GPT2Attacker(
        gpt2_model,
        gpt2_tokenizer,
        device
    )

    bert_attacker = BERTAttacker(
        AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device),
        AutoTokenizer.from_pretrained("bert-base-uncased"),
        device
    )

    attackers = {
        "t5": (t5_attacker, optimizer),
        "gpt2": (gpt_attacker, None),
        "bert": (bert_attacker, None),
    }

    # Data
    df = pd.read_csv(
       r"C:\Users\zlata\.cache\kagglehub\datasets\rtatman\deceptive-opinion-spam-corpus\versions\2\deceptive-opinion.csv"
    )
    df = df[["text", "deceptive"]].dropna()
    df["label"] = df["deceptive"].map({"truthful": 0, "deceptive": 1})

    deceptive_samples = list(enumerate(
        df[df["label"] == LABEL_DECEPTIVE]["text"].tolist()
    ))

    os.makedirs("results", exist_ok=True)

    #  Main Loop
    for attacker_name, (attacker, optimizer) in attackers.items():

        for victim_name, victim_model in victim_models.items():

            print(f"\n ===== {attacker_name} attacking → {victim_name} ===== ")

            save_path = f"results/{attacker_name}_{victim_name}.csv"

            #  Load existing results if resuming
            if os.path.exists(save_path):
                existing_df = pd.read_csv(save_path)

                processed_ids = set(
                    zip(existing_df["sample_id"],
                        existing_df["attacker"],
                        existing_df["victim"])
                )

                print(f"Resuming: {len(processed_ids)} already done")
            else:
                processed_ids = set()

            #  Always start fresh buffer 
            results = []


            attack_candidates = [
                (sid, text) for sid, text in deceptive_samples
                if (sid, attacker_name, victim_name) not in processed_ids
                and victim_model.predict(text)[0] == LABEL_DECEPTIVE
            ]

            pbar = tqdm(attack_candidates)
            
            success_count = 0
            total_count = 0
            sim_sum = 0
            sim_count = 0
            steps_sum = 0

            for i, (sample_id, review) in enumerate(pbar):

                transfer_preds = None

                attack_result = run_attack(
                    original_text=review,
                    attacker=attacker,
                    victim_model=victim_model,
                    sbert_model=sbert_model,
                    optimizer=optimizer,
                    sim_threshold=SIM_THRESHOLD
                )

                total_count += 1

                if attack_result["success"]:
                    success_count += 1
                    steps_sum += attack_result["steps"]

                    adv_text = attack_result["adversarial_text"]
                    transfer_preds = {
                        "encoder_only": encoder_only.predict(adv_text),
                        "decoder_only": decoder_only.predict(adv_text),
                        "encoder_decoder": encoder_decoder.predict(adv_text),
                    }

                    if attack_result["similarity"] is not None:
                        sim_sum += attack_result["similarity"]
                        sim_count += 1
                
                current_asr = success_count / total_count
                avg_sim = sim_sum / sim_count if sim_count > 0 else 0

                pbar.set_postfix({
                    "ASR": f"{current_asr:.2f}",
                    "sim": f"{avg_sim:.2f}"
                })

                result_row = {
                    "sample_id": sample_id,
                    "attacker": attacker_name,
                    "victim": victim_name,
                    "original_text": review,
                    "success": attack_result["success"],
                    "steps": attack_result["steps"],
                    "similarity": attack_result["similarity"],
                    "adversarial_text": attack_result.get("adversarial_text"),
                    "transfer_preds": transfer_preds,
                    "history": attack_result["history"]
                }

                results.append(result_row)

                # SAVE EVERY 100 SAMPLES 
                if (i + 1) % 100 == 0:
                    df_out = pd.DataFrame(results)

                    #  Summary stats
                    total = total_count
                    # successes = [r for r in results if r["success"]]
                    successes = success_count
                    steps_sum += attack_result["steps"]

                    asr = successes / total if total > 0 else 0
                    avg_steps = steps_sum / successes if successes > 0 else 0
                    avg_sim = sim_sum / sim_count if sim_count > 0 else 0

                    print("\n===== SUMMARY =====")
                    print(f"Attacker: {attacker_name}")
                    print(f"Victim: {victim_name}")
                    print(f"Samples: {total}")
                    print(f"ASR: {asr:.3f}")
                    print(f"Avg steps: {avg_steps:.2f}")
                    print(f"Avg similarity: {avg_sim:.3f}")
                    print(f"Tranferability: {transfer_preds}")
                    print("===================\n")

                    print(f"Final saved: {save_path}")

                    if os.path.exists(save_path):
                        df_out.to_csv(save_path, mode="a", header=False, index=False)
                    else:
                        df_out.to_csv(save_path, index=False)

                    results = []  #  clear buffer after writing


            if results:
                df_out = pd.DataFrame(results)

                if os.path.exists(save_path):
                    df_out.to_csv(save_path, mode="a", header=False, index=False)
                else:
                    df_out.to_csv(save_path, index=False)

            print(f"Final saved: {save_path}")


if __name__ == "__main__":
    main()