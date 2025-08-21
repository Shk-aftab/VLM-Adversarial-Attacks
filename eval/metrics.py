import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from torchmetrics.multimodal.clip_score import CLIPScore

def compute_bleu(references, hypotheses):
    chencherry = SmoothingFunction()
    scores = []
    for refs, hyp in zip(references, hypotheses):
        refs_tok = [nltk.word_tokenize(r.lower()) for r in refs]
        hyp_tok = nltk.word_tokenize(hyp.lower())
        scores.append(sentence_bleu(refs_tok, hyp_tok, smoothing_function=chencherry.method1))
    return sum(scores)/max(len(scores),1)

def compute_rougeL(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for refs, hyp in zip(references, hypotheses):
        best = max(scorer.score(r, hyp)['rougeL'].fmeasure for r in refs)
        scores.append(best)
    return sum(scores)/max(len(scores),1)

# Global CLIP model instance to avoid reloading
_clip_model = None

def compute_clipscore(hypotheses, images_paths, device='cuda'):
    global _clip_model
    if _clip_model is None:
        _clip_model = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16').to(device)
    
    scores = []
    with torch.no_grad():
        for hyp, img in zip(hypotheses, images_paths):
            scores.append(_clip_model(img, hyp).item())
    return sum(scores)/max(len(scores),1)
