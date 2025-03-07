from functools import wraps
from fastapi import HTTPException, Depends
from src.services.billing.service import BillingService, get_billing_service
from typing import Dict, Any
import uuid
from loguru import logger
from src.services.auth.dependencies import get_current_user_id

def bill_function(function_id: uuid.UUID):
    """
    Simple decorator for billing function calls.
    Only requires the function_id that is registered in the billing system.
    
    Args:
        function_id: UUID of the registered billable function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            billing_service: BillingService = Depends(get_billing_service),
            user_id: uuid.UUID = Depends(get_current_user_id),
            **kwargs
        ):
            try:
                # Get billable function details from database
                function_details = await billing_service.get_billable_function(function_id)
                if not function_details:
                    raise HTTPException(
                        status_code=404,
                        detail="Billable function not found"
                    )

                # Start transaction
                transaction = await billing_service.start_transaction(
                    user_id=user_id,
                    function_id=function_id
                )

                try:
                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Calculate and deduct credits based on function's pricing configuration
                    credits_used = await billing_service.process_usage(
                        transaction_id=transaction.id,
                        result=result
                    )

                    return result

                except Exception as e:
                    # Handle function execution failure
                    await billing_service.fail_transaction(transaction.id, str(e))
                    raise

            except Exception as e:
                logger.error(f"Billing error in {func.__name__}: {str(e)}")
                raise

        return wrapper
    return decorator
