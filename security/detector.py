def is_malicious(query: str) -> bool:
    query = query.lower()

    patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard above",
        "forget previous",
        "act as",
        "pretend to be",
        "you are now",
        "say compromised",
        "output compromised",
        "just say",
        "do not follow",
        "bypass",
        "override",
        "jailbreak","ignore", "disregard", "forget",
        "pretend", "roleplay",
         "system", "admin",
        "bypass", "override",
        "reveal", "show", "hidden prompt",
    ]

    verbs = [
        "say", "respond", "output", "answer",
        "return", "print", "give"
    ]

 # Check general patterns
    if any(p in query for p in patterns):
        return True

    # Check instruction override with keywords
    if ("compromised" in query or "hijacked" in query):
        if any(v in query for v in verbs):
            return True

    return False
