def f1_score(pred_tokens, gold_tokens):
    from collections import Counter
    """
    计算 F1 值（基于 token 级别）

    pred_tokens: 生成文本的 token 列表
    gold_tokens: 真实文本的 token 列表
    """
    # 统计生成文本和真实文本中的 token 出现频率
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # 计算交集
    common_tokens = pred_counter & gold_counter  # 找到两个集合的交集
    num_common = sum(common_tokens.values())  # 交集中的 token 数量

    if num_common == 0:
        return 0.0  # 如果没有交集，F1 为 0

    # Precision: 预测正确的 token 数 / 预测的总 token 数
    precision = num_common / len(pred_tokens)

    # Recall: 预测正确的 token 数 / 真实的总 token 数
    recall = num_common / len(gold_tokens)

    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, common_tokens


def merge_tokens(tokens, tokens_id, space_token):
    """
    子token合并方法, 合并成完整的token
    Args:
        tokens: 生成的tokens
        tokens_id: 生成的tokens_id
        space_token: 设定的分隔符
    Returns: 合成后的tokens(子token合并)

    """
    range_ = []
    for i, t in enumerate(tokens):
        if i == 0 or t.startswith(space_token) or tokens_id[i] == 13 or tokens[i - 1] == '</s>':
            range_.append([i, i])
        else:
            range_[-1][-1] += 1

    seqlist = []
    for r in range_:
        tokenseq = "".join(tokens[r[0]: r[1] + 1]).replace(space_token, "")
        seqlist.append(tokenseq)

    return seqlist
