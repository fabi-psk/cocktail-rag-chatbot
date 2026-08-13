from fastapi import APIRouter

from models.cocktail import ChatRequest, PreferenceRemoveRequest
from services.chat_service import build_chat_response, build_session_recommendation_response
from services.conversation_service import conversation_service


router = APIRouter()


@router.get("/chat")
async def chat_search_get(frage: str, session_id: str | None = None):
    response = await build_chat_response(frage, session_id=session_id)
    response["frage"] = frage
    return response


@router.post("/chat")
async def chat_search_post(request: ChatRequest):
    return await build_chat_response(request.message, request.history, session_id=request.session_id)


@router.delete("/chat/session/{session_id}")
def reset_chat_session(session_id: str):
    conversation_service.reset_preferences(session_id)
    return {"status": "ok"}


@router.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str):
    return await build_session_recommendation_response(session_id)


@router.post("/chat/session/{session_id}/preferences/remove")
async def remove_chat_preference(session_id: str, request: PreferenceRemoveRequest):
    conversation_service.remove_preference(session_id, request.field, request.value)
    return await build_session_recommendation_response(session_id)
