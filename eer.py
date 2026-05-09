def compute_err(raw, gold, pred, ign_caps=False, verbose=False, info=True):
    cor = 0
    changed = 0
    total = 0

    if len(gold) != len(pred):
        raise ValueError(
            f"gold has {len(gold)} sentences, but pred has {len(pred)} sentences"
        )

    for sent_raw, sent_gold, sent_pred in zip(raw, gold, pred):
        if len(sent_gold) != len(sent_pred):
            raise ValueError(
                "A sentence has different length in prediction. "
                "Check sentence/token order."
            )

        for word_raw, word_gold, word_pred in zip(sent_raw, sent_gold, sent_pred):
            if ign_caps:
                word_raw = word_raw.lower()
                word_gold = word_gold.lower()
                word_pred = word_pred.lower()

            if word_raw != word_gold:
                changed += 1

            if word_gold == word_pred:
                cor += 1
            elif verbose:
                print(word_raw, word_gold, word_pred)

            total += 1

    accuracy = cor / total
    lai = (total - changed) / total

    if changed == 0:
        err = 0.0
    else:
        err = (accuracy - lai) / (1 - lai)

    if info:
        print("Baseline acc.(LAI): {:.2f}".format(lai * 100))
        print("Accuracy:           {:.2f}".format(accuracy * 100))
        print("ERR:                {:.2f}".format(err * 100))

    return lai, accuracy, err