import logging
import requests
from datetime import datetime
from langchain.tools import tool

from configuration.config import AppConfig

logger = logging.getLogger(__name__)

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str, date: str = "latest") -> str:
    """
    Converts a specific numerical amount from one currency to another using historical or current exchange rates.
    Use this tool WHENEVER the user asks about currency conversion, exchange rates, 
    or calculating travel expenses in a different currency.
    
    Args:
        amount: The numerical amount to convert (e.g., 100.50).
        from_currency: The 3-letter currency code to convert FROM (e.g., 'EUR', 'USD', 'HUF').
        to_currency: The 3-letter currency code to convert TO (e.g., 'USD', 'GBP', 'EUR').
        date: The date of the transaction in 'YYYY-MM-DD' format. If no date is specified by the user, use 'latest'.
        
    Returns:
        A natural language string containing the converted amount, or a clear error message 
        if the API fails or the currency code/date is invalid.
    """
    # Clean and normalize input parameters
    try:
        amount = float(amount)
        from_currency = str(from_currency).upper().strip()
        to_currency = str(to_currency).upper().strip()
        date = str(date).strip().lower()
        
        # Validate date format if it's not 'latest'
        if date != "latest":
            datetime.strptime(date, "%Y-%m-%d")
            
    except ValueError as e:
        if "time data" in str(e):
            logger.error("Invalid date format provided by agent: %s", date)
            return "Error: Invalid date format. Please provide the date in YYYY-MM-DD format."
        logger.error("Invalid data types provided for currency conversion: amount=%s", amount)
        return "Error: Please provide a valid numerical amount for conversion."

    logger.info("Currency conversion requested: %f %s to %s for date: %s", amount, from_currency, to_currency, date)
    
    if from_currency == to_currency:
        return f"The amount {amount} {from_currency} is exactly the same in {to_currency}."

    # Build the API request URL using central configuration and dynamic date
    url = f"{AppConfig.FRANKFURTER_API_URL}/{date}"
    params = {
        "amount": amount,
        "from": from_currency,
        "to": to_currency
    }
    
    try:
        logger.info("Calling Frankfurter API at %s with params: %s", url, params)
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Safely extract the target currency rate
        rates = data.get("rates", {})
        converted_amount = rates.get(to_currency)
        
        if converted_amount is None:
            error_msg = f"Target currency '{to_currency}' not found in API response."
            logger.error(error_msg)
            return f"Error: The currency '{to_currency}' is not supported or not found for this date."
            
        logger.info("Successfully converted %f %s to %f %s (Date: %s)", amount, from_currency, converted_amount, to_currency, date)
        
        return f"On {date}, the conversion for {amount} {from_currency} was {converted_amount} {to_currency}."
        
    except Exception as ex:
        logger.error("Unexpected error in convert_currency tool: %s", str(ex))
        return "Error: An unexpected issue occurred during the currency conversion."

if __name__ == "__main__":
    # Local testing block
    from utils.logging import setup_logger
    setup_logger()

    logger.info("=== Starting Currency Tool Pipeline Test ===")
    
    # Test 1: Successful conversion (Latest)
    logger.info("\n--- Test 1: Valid Conversion (100 EUR to USD, latest) ---")
    result_success = convert_currency.invoke({"amount": 100, "from_currency": "EUR", "to_currency": "USD"})
    logger.info("Result 1: %s", result_success)
    
    # Test 2: Historical Conversion
    logger.info("\n--- Test 2: Historical Conversion (100 EUR to USD on 2023-01-10) ---")
    result_historical = convert_currency.invoke({"amount": 100, "from_currency": "EUR", "to_currency": "USD", "date": "2023-01-10"})
    logger.info("Result 2: %s", result_historical)
    
    # Test 3: Invalid Date Format
    logger.info("\n--- Test 3: Invalid Date Format (100 EUR to USD on 10th Jan 2023) ---")
    result_bad_date = convert_currency.invoke({"amount": 100, "from_currency": "EUR", "to_currency": "USD", "date": "10th Jan 2023"})
    logger.info("Result 3: %s", result_bad_date)
    
    logger.info("=== Currency Tool Pipeline Test Complete ===")