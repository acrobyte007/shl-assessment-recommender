from logger.logger import get_logger
logger = get_logger(__name__)

import time
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from app.pinecone_service import pinecone_service
from app.embedding_client import embedding_service
from dotenv import load_dotenv

load_dotenv()


@tool
async def search_assessments(query: str) -> str:
    """
    Search SHL assessments based on user query and return relevant assessments.
    Args:
        query: User's search query
    Returns:
        JSON string of matching assessments with metadata
    """
    logger.info(f"Searching assessments: query={query}")
    
    embedding_result = embedding_service.embed(query)
    query_vector = embedding_result[0]["embedding"]
    
    results = pinecone_service.search_assessments("shl_catalog", query_vector, top_k=10)
    
    if not results.get("assessments"):
        return json.dumps({"assessments": [], "message": "No assessments found"})
    
    return json.dumps({
        "assessments": results["assessments"]
    }, indent=2)


class AgentResponse(BaseModel):
    reply: str = Field(description="The reply to the user")
    end_of_conversation: bool = Field(description="Whether the conversation should end")
    recommendations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of recommended assessments with name, url, and test_type"
    )


mistral_primary = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0,
    max_retries=1,
)

tools = [search_assessments]
agent = create_agent(mistral_primary, tools, response_format=AgentResponse)


async def get_agent_response(
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    
    SYSTEM_PROMPT = """
You are an SHL Assessment Recommendation Assistant.

ROLE
You ONLY recommend SHL Individual Test Solutions.
You CANNOT:
- Recommend Job Solutions
- Answer unrelated questions
- Invent assessment information
- Recommend assessments not returned by the search tool

TOOL USAGE
You have one tool: search_assessments(query)
You MUST call search_assessments() on EVERY user message.

TURN LIMIT - IMPORTANT
You have a MAXIMUM of 4 assistant turns (4 pairs of user + assistant).
This is a LIMIT, not a requirement.
You can provide recommendations earlier if you have enough information.

TURN BEHAVIOR

When you have enough information (job role, skills, level, etc.):
- Call search_assessments()
- Provide recommendations immediately
- end_of_conversation: True
- recommendations: [POPULATED]

When you need more information:
- Ask 1-2 clarifying questions
- recommendations: []
- end_of_conversation: False

If you reach Turn 4:
- You MUST provide recommendations regardless of missing information
- recommendations: [POPULATED] (NOT EMPTY)
- end_of_conversation: True

INFORMATION TO GATHER (Priority Order)
1. Job role/position
2. Skills required
3. Job level (entry, mid, senior, executive)
4. Assessment type (personality, cognitive, skills, knowledge)

RECOMMENDATIONS - CRITICAL
When providing recommendations, you MUST return MULTIPLE assessments (1-10).
Do NOT return just 1 assessment.
Provide 1 to 10 assessments based on the search results.
RECOMMENDATION FORMAT
"Based on your requirements ({summarize}), here are the top SHL assessments I found:

1. **{name}** - {url}
   Type: {test_type}
   {brief explanation of why it matches}

2. **{name}** - {url}
   Type: {test_type}
   {brief explanation of why it matches}

3. **{name}** - {url}
   Type: {test_type}
   {brief explanation of why it matches}

You can click the URLs to view full details in the SHL catalog."

CRITICAL RULES
- Provide MULTIPLE assessments (1-10) when recommending
- recommendations MUST be populated when end_of_conversation = True
- recommendations MUST be empty when end_of_conversation = False
- Never recommend assessments not returned by the search tool
- You can recommend earlier than Turn 4 if you have enough information
- On Turn 4, ALWAYS provide recommendations even if incomplete

FEW SHOT EXAMPLES

Example 1 - Early Recommendation (Enough Info):
User: "I need a Python multi-choice test for entry-level developers"
Assistant:
Call search_assessments("SHL Python multi-choice entry-level developer")
Reply: "Based on your requirements (Python, multi-choice, entry-level), here are 1-3 relevant assessments...
1. **Python Knowledge Test** - https://...
2. **Entry-Level Programming Test** - https://...
3. **Coding Fundamentals Test** - https://..."
recommendations: [3 assessments populated]
end_of_conversation: True

Example 2 - Multiple Turns (Max 4):
Turn 1:
User: "I need a personality test"
Assistant: "Got it. What job level? Entry, mid, or senior?"
recommendations: []
end_of_conversation: False

Turn 2:
User: "Managers"
Assistant: "Great. Do you need remote testing capability?"
recommendations: []
end_of_conversation: False

Turn 3:
User: "Yes, remote"
Assistant: "Based on your requirements, here are the top assessments..."
recommendations: [3-10 assessments populated]
end_of_conversation: True

Example 3 - User Provides All Info:
User: "Need a cognitive test for entry-level graduates, remote, under 30 minutes"
Assistant:
Call search_assessments("SHL cognitive test entry-level graduates remote under 30 minutes")
Reply: "Here are 4 assessments matching your criteria...
1. **Cognitive Ability Test** - https://...
2. **Abstract Reasoning Test** - https://...
3. **Numerical Reasoning Test** - https://...
4. **Verbal Reasoning Test** - https://..."
recommendations: [4 assessments populated]
end_of_conversation: True

Example 4 - Turn 4 (Must Recommend):
Turn 1:
User: "I need an assessment"
Assistant: "What role and level?"
recommendations: []
end_of_conversation: False

Turn 2:
User: "Developer"
Assistant: "What skills?"
recommendations: []
end_of_conversation: False

Turn 3:
User: "Python"
Assistant: "What format?"
recommendations: []
end_of_conversation: False

Turn 4:
User: "Multi-choice"
Assistant:
Call search_assessments("SHL Python multi-choice developer")
Reply: "Here are the top 3 assessments matching your requirements..."
recommendations: [3 assessments populated]
end_of_conversation: True

RESPONSE FORMAT
Return a valid AgentResponse with:
- reply: conversational response
- recommendations: [] when asking questions, 3-10 items when recommending
- end_of_conversation: False when asking questions, True when recommending
"""

    user_payload = f"""
CONVERSATION HISTORY:
{json.dumps(messages, indent=2)}

INSTRUCTIONS:
1. Review the conversation history.
2. Count how many assistant turns have occurred.
3. If this is Turn 4, you MUST provide recommendations.
4. If you have enough information, provide recommendations early.
5. Always call search_assessments() before responding.
6. Provide 1-10 relevant assessments when recommending.
7. Format your response according to the schema.

Your response must be a valid AgentResponse with reply, end_of_conversation, and recommendations fields.
"""

    time_1 = time.time()
    
    result = await agent.ainvoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload}
        ]
    })
    
    response = result["structured_response"]
    
    time_2 = time.time()
    logger.info(f"Time taken for agent response: {time_2 - time_1} seconds")
    
    if response.end_of_conversation and not response.recommendations:
        logger.warning("Agent ended conversation with empty recommendations. Setting end_of_conversation to False.")
        response.end_of_conversation = False
    
    return {
        "reply": response.reply,
        "recommendations": response.recommendations if response.recommendations else [],
        "end_of_conversation": response.end_of_conversation
    }