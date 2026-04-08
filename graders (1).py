def grade_episode(rewards):
    total = sum(rewards)
    
    # Normalize score between 0 and 1
    score = max(0.0, min(1.0, (total + 1) / 2))
    return score
