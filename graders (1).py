def grade_episode(rewards):
    total = sum(rewards)
    
    # Normalize score between (0, 1) strictly
    score = (total + 1) / 2
    score = max(0.01, min(0.99, score))
    
    return score
