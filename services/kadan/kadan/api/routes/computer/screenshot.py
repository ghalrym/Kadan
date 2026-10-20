from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/api/computer/screenshot")
async def computer_screenshot(request: Request) -> Response:
    content, media_type = await request.app.state.services.sandbox.screenshot()
    return Response(
        content,
        media_type=media_type,
        headers={"Cache-Control": "private, no-store"},
    )
