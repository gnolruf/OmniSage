from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

class QueryRouter:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.encoder = HuggingFaceEncoder(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Define routes for different query types
        programming_route = Route(
            name="programming",
            utterances=[
                "How do I write a function in Python?",
                "What's the syntax for a for loop?",
                "How to declare variables in JavaScript?",
                "Can you explain Object-Oriented Programming?",
                "How do I use try-except blocks?",
                "What are arrays in programming?",
                "How to implement data structures?",
                "Explain recursion in programming",
                "What is inheritance in OOP?",
                "How to handle exceptions in code?"
            ]
        )
        
        # Create the route layer
        self.router = RouteLayer(
            encoder=self.encoder,
            routes=[programming_route]
        )
    
    def is_programming_question(self, query: str) -> bool:
        """
        Determine if the input query is a programming-related question.
        """
        result = self.router(query)
        if self.debug:
            print(f"Router result: {result}")
        return result.name == "programming" if result else False