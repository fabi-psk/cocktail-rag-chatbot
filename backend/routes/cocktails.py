from fastapi import APIRouter, HTTPException, Query

from repositories.cocktail_repository import CocktailRepository, CocktailRepositoryError
from services.cocktail_service import search_by_query_params


router = APIRouter()
repository = CocktailRepository()


@router.get("/cocktails")
def get_cocktails():
    try:
        return repository.list_all()
    except CocktailRepositoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/cocktails/{name}")
def get_cocktail_by_name(name: str):
    try:
        cocktail = repository.find_by_name(name)
    except CocktailRepositoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if cocktail is None:
        raise HTTPException(status_code=404, detail="Cocktail nicht gefunden")
    return cocktail


@router.get("/search")
def search_cocktails_endpoint(
    spirituose: str | None = Query(default=None),
    geschmack: str | None = Query(default=None),
    kategorie: str | None = Query(default=None),
    staerke: str | None = Query(default=None),
):
    try:
        result = search_by_query_params(spirituose, geschmack, kategorie, staerke)
    except CocktailRepositoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "anzahl": len(result),
        "cocktails": result,
    }
