from fastapi import FastAPI
from langserve import add_routes
from src.graph import graph  # Import your graph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Sales Assistant API",
    description="A proactive sales assistant that helps manage deals and tasks",
    version="1.0",
)

# Add routes for the sales assistant
add_routes(
    app,
    graph,
    path="/sales-assistant",
    input_type=dict,  # Input will be a dictionary with user_id and message
)

# Add health check
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)