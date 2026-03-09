from rag import answer_question

result = answer_question("What is Tony's favorite color?", k=6)
print(result["answer"])
