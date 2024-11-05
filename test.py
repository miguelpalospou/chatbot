import requests

def chat_with_assistant(user_id: str, message: str = ""):
    response = requests.post(
        "http://localhost:8000/sales-assistant/invoke",
        json={
            "input": {
                "user_id": user_id,
                "last_user_message": message,
                "conversation_history": [],
                "pending_insights": [],
                "aws_data": {},
                "waiting_for_response": False,
                "current_topic": "",
                "topic_addressed": False
            }
        }
    )
    return response.json()


# Example usage
if __name__ == "__main__":
    # Start conversation
    response = chat_with_assistant("12345")