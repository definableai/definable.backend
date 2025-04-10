"""Module for email service."""  # noqa: A005

import resend
from fastapi import HTTPException

from config.settings import settings

resend.api_key = settings.resend_api_key


async def send_invitation_email(
  email: str,
  organization_name: str,
  accept_url: str,
  reject_url: str,
) -> None:
  """Send invitation email to new team member."""
  try:
    params: resend.Emails.SendParams = {
      "from": "Team Invites <invites@dolbo.ai>",
      "to": email,
      "subject": f"You've been invited to join {organization_name}",
      "html": f"""
                  <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                      <h1 style="color: #2ecc71;">Welcome to {organization_name}!</h1>
                      <p>You've been invited to join our team. Please choose to accept or decline the invitation:</p>

                      <div style="margin: 30px 0;">
                          <a href="{accept_url}"
                              style="display: inline-block; padding: 12px 24px; background-color: #2ecc71; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px;">
                              Accept Invitation
                          </a>
                          <a href="{reject_url}"
                              style="display: inline-block; padding: 12px 24px; background-color: #e74c3c; color: white; text-decoration: none; border-radius: 4px;">
                              Decline Invitation
                          </a>
                      </div>

                      <p style="color: #666; font-size: 14px;">
                          This invitation link will expire in 48 hours.<br>
                          If you didn't request this invitation, you can safely ignore this email.
                      </p>
                  </div>
              """,  # noqa: E501
    }
    resend.Emails.send(params)
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to send invitation email: {str(e)}")


async def send_password_reset_email(email: str, reset_token: str) -> None:
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


async def send_password_reset_confirmation_email(email: str) -> None:
  """Send password reset confirmation email."""
  try:
    params: resend.Emails.SendParams = {
      "from": "Password Reset <noreply@dolbo.ai>",
      "to": email,
      "subject": "Password Reset Successful",
      "html": """
                  <h1>Password Reset Successful</h1>
                  <p>Your password has been successfully reset.</p>
                  <p>If you did not make this change, please contact our support team immediately.</p>
                  <p>For security reasons, we recommend reviewing your account activity.</p>
              """,
    }
    resend.Emails.send(params)
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to send password reset confirmation email: {str(e)}")
