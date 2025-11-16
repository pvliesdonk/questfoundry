"""User settings endpoints for BYOK management

This router provides endpoints for users to manage their provider API keys.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from questfoundry.providers.config import ProviderConfig

from ..user_settings import get_user_provider_config, save_user_provider_config

router = APIRouter(prefix="/user", tags=["user_settings"])


class UserSettingsResponse(BaseModel):
    """Response containing user settings"""

    user_id: str = Field(..., description="User identifier")
    has_openai_key: bool = Field(..., description="Whether OpenAI key is configured")
    has_anthropic_key: bool = Field(
        ..., description="Whether Anthropic key is configured"
    )
    has_google_key: bool = Field(..., description="Whether Google key is configured")


class ProviderKeysRequest(BaseModel):
    """Request to update provider keys"""

    openai_api_key: str | None = Field(None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(None, description="Anthropic API key")
    google_api_key: str | None = Field(None, description="Google API key")


class ProviderKeysResponse(BaseModel):
    """Response after updating provider keys"""

    status: str = Field(..., description="Update status")
    message: str = Field(..., description="Status message")


@router.get("/settings", response_model=UserSettingsResponse)
async def get_user_settings(
    request: Request,
) -> UserSettingsResponse:
    """
    Get user settings.

    Returns information about which provider keys are configured,
    without revealing the actual keys.

    Args:
        request: FastAPI request (contains user_id)

    Returns:
        User settings information

    Raises:
        HTTPException: 500 if query fails
    """
    user_id = request.state.user_id

    try:
        # Get user's provider config
        config = await get_user_provider_config(user_id)

        return UserSettingsResponse(
            user_id=user_id,
            has_openai_key=config.openai_api_key is not None,
            has_anthropic_key=config.anthropic_api_key is not None,
            has_google_key=config.google_api_key is not None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user settings: {str(e)}",
        ) from e


@router.put("/settings/keys", response_model=ProviderKeysResponse)
async def update_provider_keys(
    request: Request,
    keys_request: ProviderKeysRequest,
) -> ProviderKeysResponse:
    """
    Update provider API keys (BYOK).

    Keys are encrypted using Fernet symmetric encryption before storage.
    Only non-None keys are updated; None values leave existing keys unchanged.

    Args:
        request: FastAPI request (contains user_id)
        keys_request: Provider keys to update

    Returns:
        Update status

    Raises:
        HTTPException: 500 if update fails
    """
    user_id = request.state.user_id

    try:
        # Get existing config
        config = await get_user_provider_config(user_id)

        # Update only provided keys
        if keys_request.openai_api_key is not None:
            config.openai_api_key = keys_request.openai_api_key

        if keys_request.anthropic_api_key is not None:
            config.anthropic_api_key = keys_request.anthropic_api_key

        if keys_request.google_api_key is not None:
            config.google_api_key = keys_request.google_api_key

        # Save encrypted config
        await save_user_provider_config(user_id, config)

        return ProviderKeysResponse(
            status="success",
            message="Provider keys updated successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update provider keys: {str(e)}",
        ) from e


@router.delete("/settings/keys", response_model=ProviderKeysResponse)
async def delete_provider_keys(
    request: Request,
) -> ProviderKeysResponse:
    """
    Delete all provider API keys.

    This removes all BYOK configuration for the user.

    Args:
        request: FastAPI request (contains user_id)

    Returns:
        Deletion status

    Raises:
        HTTPException: 500 if deletion fails
    """
    user_id = request.state.user_id

    try:
        # Save empty config
        empty_config = ProviderConfig()
        await save_user_provider_config(user_id, empty_config)

        return ProviderKeysResponse(
            status="success",
            message="All provider keys deleted",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete provider keys: {str(e)}",
        ) from e
