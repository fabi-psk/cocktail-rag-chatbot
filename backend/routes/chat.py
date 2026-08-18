from fastapi import APIRouter, HTTPException

from models.cocktail import ChatRequest, UnlockOrderingRequest
from services.chat_service import build_chat_response
from services.conversation_service import conversation_service
from services.home_recipe_service import build_home_recipe_response


router = APIRouter()


@router.get("/chat")
async def chat_search_get(frage: str, session_id: str | None = None):
    response = await build_chat_response(frage, session_id=session_id)
    response["frage"] = frage
    response["ordering_blocked"] = conversation_service.is_ordering_blocked(session_id)
    return response


@router.post("/chat")
async def chat_search_post(request: ChatRequest):
    if request.mode == "home":
        return await build_home_recipe_response(request.message, request.history)
    response = await build_chat_response(
        request.message, request.history, session_id=request.session_id
    )
    response["ordering_blocked"] = conversation_service.is_ordering_blocked(
        request.session_id
    )
    return response


@router.delete("/chat/session/{session_id}")
def reset_chat_session(session_id: str):
    conversation_service.reset_session(session_id)
    return {"status": "ok"}


@router.get("/chat/session/{session_id}/ordering-status")
def ordering_status(session_id: str):
    return {
        "ordering_blocked": conversation_service.is_ordering_blocked(session_id)
    }


@router.post("/chat/session/{session_id}/unlock-ordering")
def unlock_ordering(session_id: str, request: UnlockOrderingRequest):
    if not conversation_service.unlock_ordering(session_id, request.password):
        raise HTTPException(status_code=403, detail="Passwort ist nicht korrekt.")
    return {"ordering_blocked": False}
