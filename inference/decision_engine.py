from inference.schemas import PredictionResult, SENTIMENT_MAP
from inference.model_loader import predict, device


def decide(texts, twitter_models, amazon_models):
    """Classify texts using both Twitter and Amazon models, returning an ensemble PredictionResult per text."""
    tokenizer_tw, model_tw = twitter_models
    tokenizer_am, model_am = amazon_models

    preds_tw, probs_tw = predict(texts, tokenizer_tw, model_tw)
    preds_am, probs_am = predict(texts, tokenizer_am, model_am)

    results = []
    for i in range(len(texts)):
        p_tw, p_am = int(preds_tw[i]), int(preds_am[i])
        conf_tw = float(probs_tw[i][p_tw])
        conf_am = float(probs_am[i][p_am])

        # When both models agree, pick the one with higher confidence
        if p_tw == p_am:
            if conf_tw >= conf_am:
                final_label = p_tw
                confidence = conf_tw
                model_used = "twitter"
            else:
                final_label = p_am
                confidence = conf_am
                model_used = "amazon"
        else:
            # When models disagree, use the more conservative (lower) label
            final_label = min(p_tw, p_am)
            confidence = conf_tw if p_tw == final_label else conf_am
            model_used = "twitter" if p_tw == final_label else "amazon"

        results.append(PredictionResult(
            sentiment=SENTIMENT_MAP[final_label],
            label=final_label,
            confidence=round(confidence, 4),
            model_used=model_used
        ))

    return results
