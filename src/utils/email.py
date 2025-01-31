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
