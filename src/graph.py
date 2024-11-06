from typing import Dict, List, TypedDict
import boto3
from decimal import Decimal
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
from langgraph.graph import StateGraph, Graph, END
OPENAI_API_KEY="sk-tp-3vC_nmP5nt0_-GkyRO9r1KHsW8Yo1MF65tkeNhcT3BlbkFJl5CsVh1FpIqjDNJO85QcZLlu_NFy1CbEfkNIR7EIUA"
# Load environment variables
load_dotenv()
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
#os.environ["LANGCHAIN_API_KEY"]=str(os.getenv("LANGCHAIN_API_KEY"))
#os.environ["LANGCHAIN_ENDPOINT"]="https://eu.api.smith.langchain.com"
#os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
#os.environ["LANGCHAIN_TRACING_V2"]="true"
#os.environ["LANGCHAIN_PROJECT"]="default"

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class SalesState(TypedDict, total=False):
    """Type definition for the sales assistant state"""
    user_id: str
    conversation_history: List[Dict[str, str]]  # List of {'role': str, 'content': str}
    pending_insights: List[str]
    aws_data: Dict
    last_user_message: str
    waiting_for_response: bool
    current_topic: str  # Added to track current discussion topic
    topic_addressed: bool  # Added to track if current topic was addressed

class AWSData:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.deals_table = self.dynamodb.Table('Deals')
        self.tasks_table = self.dynamodb.Table('Tasks')
        
    async def get_user_data(self, user_id: str) -> Dict:
        """Fetch all relevant user data from AWS DynamoDB"""
        deals = self.deals_table.query(
            KeyConditionExpression='UserID = :uid',
            ExpressionAttributeValues={':uid': user_id}
        )
        
        tasks = self.tasks_table.query(
            KeyConditionExpression='UserID = :uid',
            ExpressionAttributeValues={':uid': user_id}
        )
        
        return {
            'deals': deals['Items'],
            'tasks': tasks['Items']
        }

# Initialize services at module level
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
aws_client = AWSData()

def determine_next_step(state: SalesState) -> str:
    """Simple decision tree with safe key access"""
    # Use .get() with default values for safe access
    waiting_for_response = state.get('waiting_for_response', False)
    last_user_message = state.get('last_user_message', '')
    current_topic = state.get('current_topic', '')
    topic_addressed = state.get('topic_addressed', False)
    pending_insights = state.get('pending_insights', [])

    if waiting_for_response:
        return "end"  # Wait for user input
        
    if last_user_message:
        # If we have a current topic and it's not addressed, continue with it
        if current_topic and not topic_addressed:
            return "continue_current_topic"
        else:
            return "handle_user_question"
    
    # If we're not waiting and have pending insights, share next one
    if pending_insights and not waiting_for_response:
        return "share_next_insight"
        
    return "end"





async def initialize_session(state: SalesState) -> SalesState:
    """Get data and generate list of insights"""

        # Ensure state is a dictionary
    if not isinstance(state, dict):
        state = {}
    
    user_id = state.get('user_id', '')

    # Create fresh state with defaults
    initial_state = {
        'user_id': user_id,
        'conversation_history': [],
        'pending_insights': [],
        'aws_data': {},
        'last_user_message': '',
        'waiting_for_response': False,
        'current_topic': '',
        'topic_addressed': False
    }

    aws_data = await aws_client.get_user_data(initial_state['user_id'])

    messages = [
        SystemMessage(content="""You are a proactive sales assistant. Review the deals and tasks and list out 
        the most important things that need attention in a natural way and directed to the sales person, in order of priority. Consider:
        - Urgent deadlines
        - Missing information
        - Opportunities to increase win probability
        - Important reminders
        
        List them in natural language, don't include the order in the list like "1...." just use natural language, ordered by importance. Each item should include what needs attention 
        and 1-2 questions you'd want to ask about it."""),
        HumanMessage(content=f"User data: {json.dumps(aws_data, cls=DecimalEncoder)}")
    ]
    
    response = await llm.ainvoke(messages)
    insights = [insight.strip() for insight in response.content.split('\n\n') if insight.strip()]
    
    return {
        **state,
        'aws_data': aws_data,
        'pending_insights': insights,
        'waiting_for_response': False,
        'current_topic': '',
        'topic_addressed': False
    }

async def share_next_insight(state: SalesState) -> SalesState:
    """Share the next insight proactively"""

    # Safely access all state keys with defaults
    pending_insights = state.get('pending_insights', [])
    waiting_for_response = state.get('waiting_for_response', False)
    current_topic = state.get('current_topic', '')
    topic_addressed = state.get('topic_addressed', False)
    conversation_history = state.get('conversation_history', [])



    if (pending_insights and 
        not waiting_for_response and 
        (not current_topic or topic_addressed)):
        
        next_insight = pending_insights[0]


        messages = [
            SystemMessage(content="""Review previous conversation and share the next important topic.
            Ask a specific question to start the discussion.
            Make your message engaging and naturally conversational not just a bullet point"""),
            HumanMessage(content=f"""
            Next topic to discuss: {next_insight}
            Previous conversation: {conversation_history[-3:] if conversation_history else 'Starting conversation'}""")
        ]
        response = await llm.ainvoke(messages)
        
        return {
            **state,
            'conversation_history': [*conversation_history, {
                'role': 'assistant',
                'content': response.content
            }],
            'current_topic': next_insight,
            'topic_addressed': False,
            'pending_insights': pending_insights[1:],
            'waiting_for_response': True
        }
    return state


async def continue_current_topic(state: SalesState) -> SalesState:
    """Follow up on current topic based on user's response"""
    conversation_history = state.get('conversation_history', [])
    current_topic = state.get('current_topic', '')
    last_user_message = state.get('last_user_message', '')
    pending_insights = state.get('pending_insights', [])

    recent_messages = conversation_history[-5:]
    conversation_context = "\n".join([
        f"{'Assistant' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
        for msg in recent_messages
    ])
    messages = [
        SystemMessage(content="""You are a sales assistant. Based on the user's response about the current topic and the conversation context, determine if:
        1. If you need more information or clarification:
            - Ask specific follow-up questions about the topic if needed
            - Focus on getting actionable information
            - DO NOT include "TOPIC_COMPLETE"
        
        2. If the topic has been adequately addressed:
           - Acknowledge their response and summarize the key points
           - Add "TOPIC_COMPLETE" at the end
           - Provide a smooth transition
        Respond accordingly.
                      Your response should feel natural and conversational"""
            ),
        HumanMessage(content=f"""
        Current topic: {current_topic}
        Recent conversation:
        {conversation_context}
        User's latest response: {last_user_message}
        """)
    ]
    
    response = await llm.ainvoke(messages)
    clean_response = response.content.replace("TOPIC_COMPLETE", "").strip()

    # Check if topic is fully addressed
    topic_complete = "TOPIC_COMPLETE" in response.content
    if topic_complete and pending_insights:
        clean_response += "\n\nLet's talk about another important matter I noticed..."
    
    return {
        **state,
        'conversation_history': [*conversation_history, {
            'role': 'assistant',
            'content': clean_response
        }],
        'last_user_message': '',
        'topic_addressed': topic_complete,
        'waiting_for_response': True
    }





async def handle_user_question(state: SalesState) -> SalesState:
    """Handle off-topic questions with full conversation context"""
    conversation_history = state.get('conversation_history', [])
    current_topic = state.get('current_topic', '')
    last_user_message = state.get('last_user_message', '')
    aws_data = state.get('aws_data', {})

    recent_messages = conversation_history[-5:]
    conversation_context = "\n".join([
        f"{'Assistant' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
        for msg in recent_messages
    ])

    messages = [
        SystemMessage(content="""You are a sales assistant with access to the full conversation history. 
        Answer the user's question using natural language pulling information from:
        1. The conversation context
        2. Their deals and tasks data
        3. The current topic being discussed
        
        Reference previous parts of the conversation if relevant.
        After answering, smoothly transition back to the current topic if appropriate."""),
        HumanMessage(content=f"""
        Recent conversation:
        {conversation_context}
        
        Current topic: {current_topic}
        Question: {last_user_message}
        Context: {json.dumps(aws_data, cls=DecimalEncoder)}
        """)
    ]
    
    response = await llm.ainvoke(messages)
    
    return {
        **state,
        'conversation_history': [*conversation_history, {
            'role': 'assistant',
            'content': response.content
        }],
        'last_user_message': '',
        'waiting_for_response': True
    }

# Create workflow graph at module level
workflow = StateGraph(SalesState)

# Add nodes for different modes
workflow.add_node("initialize_session", initialize_session)
workflow.add_node("share_next_insight", share_next_insight)
workflow.add_node("continue_current_topic", continue_current_topic)
workflow.add_node("handle_user_question", handle_user_question)

# Add edges
for node in ["initialize_session", "share_next_insight", "continue_current_topic", "handle_user_question"]:
    workflow.add_conditional_edges(
        node,
        determine_next_step,
        {
            "share_next_insight": "share_next_insight",
            "continue_current_topic": "continue_current_topic",
            "handle_user_question": "handle_user_question",
            "end": END
        }
    )

workflow.set_entry_point("initialize_session")

# This MUST be at module level for LangGraph Studio
graph = workflow.compile()