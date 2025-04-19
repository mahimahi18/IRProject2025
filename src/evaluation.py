import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
import torch
import warnings
import re

def preprocess_text(text):
    """Clean and tokenize text"""
    if not isinstance(text, str):
        return ""
    # Basic cleaning
    text = re.sub(r'\s+', ' ', str(text).strip())
    # Simple sentence tokenization (split by periods, question marks, exclamation points)
    sentences = re.split(r'[.!?]+', text)
    # Remove empty sentences and split into words
    return [s.strip().split() for s in sentences if s.strip()]

def evaluate(log_file="src\logs\og_questions_with_answers.csv", batch_size=16, save_results=None):
    try:
        # Try different encodings to handle the file
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        data = None
        
        for encoding in encodings_to_try:
            try:
                print(f"Trying to read CSV with {encoding} encoding...")
                data = pd.read_csv(log_file, encoding=encoding)
                print(f"Successfully read the file with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding} encoding: {str(e)}")
                continue
        
        if data is None:
            print("Failed to read the CSV file with any of the attempted encodings.")
            return
            
        if "reference" not in data.columns:
            print("No reference answers available. Please add them manually.")
            return
        
        # Filter out rows with missing values
        data = data.dropna(subset=["answer", "reference"])
        
        if len(data) == 0:
            print("No valid data pairs found after filtering.")
            return
            
        print(f"Evaluating {len(data)} answer-reference pairs...")
        
        hypotheses = data["answer"].tolist()
        references = data["reference"].tolist()
        
        results = {"bleu": None, "rouge": None, "bertscore": None}
        
        # Replace just the BLEU calculation part in your script with this corrected version
# BLEU score with proper tokenization and smoothing
        try:
            smoothie = SmoothingFunction().method1
            tokenized_hyps = []
            tokenized_refs = []
            
            for hyp, ref in zip(hypotheses, references):
                if isinstance(hyp, str) and isinstance(ref, str) and hyp.strip() and ref.strip():
                    # For BLEU, we need lists of tokens, not lists of sentences
                    hyp_tokens = [token for sent in preprocess_text(hyp) for token in sent]
                    ref_tokens = [token for sent in preprocess_text(ref) for token in sent]
                    
                    if hyp_tokens and ref_tokens:  # Ensure neither is empty
                        tokenized_hyps.append(hyp_tokens)
                        tokenized_refs.append([ref_tokens])  # corpus_bleu expects a list of list of references
            
            if tokenized_hyps and tokenized_refs:
                bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)
                results["bleu"] = bleu_score
                print(f"BLEU Score: {bleu_score:.4f}")
            else:
                print("Warning: Could not calculate BLEU score - no valid text pairs after tokenization")
                    
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
        
        # ROUGE scores with error handling
        try:
            rouge = Rouge()
            # Remove empty strings and handle errors
            valid_hyps = []
            valid_refs = []
            
            for hyp, ref in zip(hypotheses, references):
                if isinstance(hyp, str) and isinstance(ref, str) and hyp.strip() and ref.strip():
                    valid_hyps.append(hyp)
                    valid_refs.append(ref)
            
            if valid_hyps and valid_refs:
                rouge_scores = rouge.get_scores(valid_hyps, valid_refs, avg=True)
                results["rouge"] = rouge_scores
                print(f"ROUGE Scores: {rouge_scores}")
            else:
                print("Warning: Could not calculate ROUGE scores - no valid text pairs")
                
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
        
        # BERTScore with batch processing
        try:
            from bert_score import score
            
            # Process in batches
            all_p_scores = []
            all_r_scores = []
            all_f1_scores = []
            
            # Remove empty strings
            valid_pairs = [(h, r) for h, r in zip(hypotheses, references) 
                          if isinstance(h, str) and isinstance(r, str) and h.strip() and r.strip()]
            
            if not valid_pairs:
                print("Warning: Could not calculate BERTScore - no valid text pairs")
            else:
                valid_hyps, valid_refs = zip(*valid_pairs)
                
                for i in range(0, len(valid_hyps), batch_size):
                    batch_hyps = valid_hyps[i:i+batch_size]
                    batch_refs = valid_refs[i:i+batch_size]
                    
                    with torch.no_grad():
                        P, R, F1 = score(batch_hyps, batch_refs, lang="en", verbose=False)
                        
                    all_p_scores.extend(P.tolist())
                    all_r_scores.extend(R.tolist())
                    all_f1_scores.extend(F1.tolist())
                
                bert_results = {
                    "precision": np.mean(all_p_scores),
                    "recall": np.mean(all_r_scores),
                    "f1": np.mean(all_f1_scores)
                }
                
                results["bertscore"] = bert_results
                print(f"BERTScore Results:")
                print(f"  Precision: {bert_results['precision']:.4f}")
                print(f"  Recall: {bert_results['recall']:.4f}")
                print(f"  F1: {bert_results['f1']:.4f}")
        
        except ImportError:
            print("BERTScore evaluation requires the bert_score package.")
            print("Install it with: pip install bert-score")
        
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
        
        # Save results if requested
        if save_results:
            try:
                import json
                with open(save_results, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {save_results}")
            except Exception as e:
                print(f"Error saving results: {e}")
                
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None

if __name__ == "__main__":
    evaluate(log_file="src\logs\og_questions_with_answers.csv", save_results="evaluation_results.json")
