from functools import wraps
from fastapi import HTTPException, Depends
from datetime import datetime
from loguru import logger
from src.dependencies.security import JWTBearer
from uuid import UUID
from src.services.billing.middleware import get_current_user_id


async def track_request_usage(
    user_id: UUID, org_id: UUID, service_name: str, session, billing_service
):
    """
    Helper function to track API usage without importing BillingService.
    This avoids circular imports.
    """
    try:
        logger.info(
            f"track_request_usage: Started for user_id={user_id}, service={service_name}"
        )

        # Get service details (globally)
        logger.info(f"track_request_usage: Getting service details for {service_name}")
        service = await billing_service._get_service_by_name(service_name, session)
        if not service:
            logger.error(f"track_request_usage: Service not found: {service_name}")
            # Instead of raising an error, we'll create a fallback service to avoid breaking the flow
            from .model import ServiceModel

            logger.info(
                f"track_request_usage: Creating temporary service object for {service_name}"
            )
            # Create a temporary service with minimal credit cost
            service = ServiceModel(
                id=UUID(
                    "00000000-0000-0000-0000-000000000000"
                ),  # Use a placeholder UUID
                name=service_name,
                credit_cost=1,  # Set a minimal cost
                description=f"Auto-generated service for {service_name}",
            )
            logger.info(
                f"track_request_usage: Created temporary service with credit_cost={service.credit_cost}"
            )

        # Check credit balance
        logger.info(
            f"track_request_usage: Checking credit balance for user_id={user_id}"
        )
        balance = await billing_service._get_or_create_balance(user_id, session)
        if balance.balance < service.credit_cost:
            logger.warning(
                f"track_request_usage: Insufficient credits for user_id={user_id}. Required: {service.credit_cost}, Available: {balance.balance}"
            )
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient credits. Required: {service.credit_cost}, Available: {balance.balance}",
            )

        logger.info(
            f"track_request_usage: Completed successfully for service={service_name}"
        )
        return service
    except Exception as e:
        logger.error(
            f"track_request_usage: Error processing request: {str(e)}", exc_info=True
        )
        # Re-raise with details for debugging
        raise HTTPException(
            status_code=500, detail=f"Error tracking request usage: {str(e)}"
        )


def track_request(service_name: str):
    """
    Decorator for tracking service usage and managing credits.

    Args:
        service_name: Name of the service being used
    """
    logger.info(f"track_request: Registering decorator for service={service_name}")

    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            **kwargs,
        ):
            try:
                logger.info(
                    f"Executing {func.__name__} with service tracking for {service_name}"
                )

                # Execute the main function first to ensure it always runs
                logger.info(
                    f"track_request: Executing wrapped function {func.__name__} first"
                )
                result = await func(*args, **kwargs)

                # Now attempt to handle billing, but don't affect the main function result
                try:
                    # Get necessary dependencies
                    # For BillingService methods, 'self' is the first argument
                    # and should be used as the billing_service
                    billing_service = None
                    if (
                        args
                        and hasattr(args[0], "_deduct_credits")
                        and hasattr(args[0], "_get_service_by_name")
                    ):
                        billing_service = args[0]
                        logger.info("track_request: Using self as billing_service")
                    else:
                        billing_service = kwargs.get("billing_service")

                    user = kwargs.get("user")
                    session = kwargs.get("session")

                    # Log dependency presence or absence
                    logger.info(
                        f"track_request: Dependencies check - billing_service: {billing_service is not None}, user: {user is not None}, session: {session is not None}"
                    )

                    if not (billing_service and user and session):
                        # If any dependencies are missing, just return the original result
                        logger.info(
                            f"track_request: Missing dependencies, skipping billing"
                        )
                        return result

                    user_id = UUID(user["id"])
                    org_id = UUID(user["org_id"])  # Get org_id from JWT token
                    logger.info(
                        f"track_request: Processing request for user_id={user_id}, org_id={org_id}"
                    )

                    # Directly deduct credits without service lookup
                    logger.info(
                        f"track_request: About to deduct 1 credit for user_id={user_id}"
                    )
                    try:
                        await billing_service._deduct_credits(user_id, 1, session)
                        logger.info(f"track_request: Credits deducted successfully")

                        # Try to track service usage if possible
                        try:
                            # Create a dummy service_id if needed
                            service_id = UUID("00000000-0000-0000-0000-000000000000")
                            logger.info(
                                f"track_request: About to track service usage for org_id={org_id}"
                            )
                            await billing_service.track_service_usage(
                                org_id, user_id, service_id, 1, session
                            )
                            logger.info(
                                f"track_request: Service usage tracked successfully"
                            )
                        except Exception as usage_e:
                            logger.error(
                                f"track_request: Error tracking usage: {str(usage_e)}"
                            )
                    except Exception as deduct_e:
                        logger.error(
                            f"track_request: Error deducting credits: {str(deduct_e)}"
                        )

                except Exception as billing_e:
                    logger.error(
                        f"track_request: Error in billing process: {str(billing_e)}",
                        exc_info=True,
                    )

                # Return the original result regardless of billing success/failure
                return result

            except Exception as e:
                logger.error(
                    f"track_request: Unhandled error in decorator: {str(e)}",
                    exc_info=True,
                )
                # As a last resort, try to execute the function directly
                return await func(*args, **kwargs)

        return wrapper

    return decorator
