print("Welcome to my computer quiz!")

playing = input("Do you want to play? ")

if playing.lower() != "yes":
    quit()

print("Okay! Let's play :)\n")

# Ask for difficulty level
difficulty = input("Choose difficulty (easy/hard): ").lower()

# Easy questions
easy_questions = [
    {
        "question": "What does CPU stand for?",
        "answer": "central processing unit"
    },
    {
        "question": "What animal is known as the 'King of the Jungle'?",
        "answer": "lion"
    },
    {
        "question": "Which planet in our solar system is famous for its beautiful rings?",
        "answer": "saturn"
    },
    {
        "question": "What is the only fruit that wears its seeds on the outside?",
        "answer": "strawberry"
    },
    {
        "question": "In the game of Monopoly, what color are the most expensive properties?",
        "answer": "dark blue"
    },
    {
        "question": "What's the name of the snowman in Disney's Frozen?",
        "answer": "olaf"
    },
    {
        "question": "Which U.S. state is known as the 'Sunshine State'?",
        "answer": "florida"
    },
    {
        "question": "What is the tallest animal in the world?",
        "answer": "giraffe"
    }
]

# Hard questions
hard_questions = [
    {
        "question": "What revolutionary HIV treatment was named the 2024 Breakthrough of the Year by Science magazine?",
        "answer": "lenacapavir - an injectable treatment that provides long-lasting protection"
    },
    {
        "question": "What newly discovered 'super-Earth' planet sits in the habitable zone 137 light-years away from us?",
        "answer": "toi-715 b"
    },
    {
        "question": "What 12-sided ancient Roman object, recently discovered in England, remains one of archaeology's biggest mysteries?",
        "answer": "roman dodecahedron"
    },
    {
        "question": "Which animated sequel became the highest-grossing film worldwide in 2024, earning over $1.6 billion?",
        "answer": "inside out 2"
    },
    {
        "question": "Taylor Swift announced her album 'The Tortured Poets Department' at which major 2024 awards ceremony?",
        "answer": "the 2024 grammy awards"
    },
    {
        "question": "Which dance style made its Olympic debut in Paris 2024 but won't appear in the 2028 games?",
        "answer": "breakdancing"
    },
    {
        "question": "In what unique location did the 2024 Paris Olympics hold their opening ceremony, breaking from tradition?",
        "answer": "on the river seine"
    },
    {
        "question": "What rare lunar event will occur on August 31, 2025?",
        "answer": "a blue moon"
    },
    {
        "question": "Archaeologists discovered ruins of an ancient lost city in Ecuador in 2024 that's estimated to be how old?",
        "answer": "around 2,500 years old"
    }
]

# Select questions based on difficulty
questions = easy_questions if difficulty == "easy" else hard_questions

# Initialize score
score = 0

# Ask questions
for i, q in enumerate(questions, 1):
    answer = input(f"Question {i}:\n{q['question']} ")
    if answer.lower() == q['answer']:
        print("Correct!")
        score += 1
    else:
        print(f"Incorrect! The answer was: {q['answer']}")

# Show final score
print(f"\nGame Over! Your score: {score}/{len(questions)}") 

