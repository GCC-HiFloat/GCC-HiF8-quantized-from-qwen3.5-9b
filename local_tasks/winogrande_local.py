def doc_to_text(doc):
    return doc["sentence"].split("_", 1)[0]

def doc_to_target(doc):
    return {"1": 0, "2": 1}[str(doc["answer"])]

def doc_to_choice(doc):
    return [doc["option1"], doc["option2"]]
