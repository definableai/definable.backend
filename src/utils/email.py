"""Module for email service."""  # noqa: A005

import resend
from fastapi import HTTPException

from config.settings import settings


class EmailUtil:
  """Email service for sending emails."""

  def __init__(self):
    resend.api_key = settings.resend_api_key

  async def send_invitation_email(self, email: str, username: str, password: str, team_name: str) -> None:
    """Send invitation email to new team member."""
    try:
      params: resend.Emails.SendParams = {
        "from": "Team Invites <invites@dolbo.ai>",
        "to": email,
        "subject": f"You've been invited to join {team_name}",
        "html": f"""
                    <h1>Welcome to {team_name}!</h1>
                    <p>You've been invited to join the team. Here are your login credentials:</p>
                    <p><strong>Username:</strong> {username}</p>
                    <p><strong>Password:</strong> {password}</p>
                    <p>Please login and change your password immediately.</p>
                    <p><a href="{settings.frontend_url}/login">Click here to login</a></p>
                """,
      }
      resend.Emails.send(params)
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to send invitation email: {str(e)}")

  async def send_password_reset_email(self, email: str, reset_token: str) -> None:
    """Send password reset email."""
    try:
      params: resend.Emails.SendParams = {
        "from": "Password Reset <noreply@dolbo.ai>",
        "to": email,
        "subject": "Password Reset Request",
        "html": f"""
                    <h1>Password Reset Request</h1>
                    <p>You have requested to reset your password. Click the link below to proceed:</p>
                    <p><a href="{settings.frontend_url}/reset-password?token={reset_token}">Reset Password</a></p>
                    <p>If you did not request this, please ignore this email.</p>
                    <p>This link will expire in 1 hour.</p>
                """,
      }
      resend.Emails.send(params)
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to send password reset email: {str(e)}")
