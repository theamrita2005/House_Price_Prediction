"""
Currency conversion and formatting module for House Price Prediction.
"""

CURRENCIES = {
    "USD": {"symbol": "$", "rate": 1.0, "name": "USD ($) - US Dollar"},
    "EUR": {"symbol": "€", "rate": 0.92, "name": "EUR (€) - Euro"},
    "GBP": {"symbol": "£", "rate": 0.79, "name": "GBP (£) - British Pound"},
    "INR": {"symbol": "₹", "rate": 83.50, "name": "INR (₹) - Indian Rupee"},
    "CAD": {"symbol": "CA$", "rate": 1.36, "name": "CAD (CA$) - Canadian Dollar"},
    "AUD": {"symbol": "A$", "rate": 1.52, "name": "AUD (A$) - Australian Dollar"},
    "JPY": {"symbol": "¥", "rate": 155.00, "name": "JPY (¥) - Japanese Yen"}
}

def convert_from_usd(amount_usd, currency_code="USD"):
    """Converts amount from USD to specified target currency."""
    info = CURRENCIES.get(currency_code, CURRENCIES["USD"])
    return amount_usd * info["rate"]

def convert_to_usd(amount, currency_code="USD"):
    """Converts amount from specified currency back to USD."""
    info = CURRENCIES.get(currency_code, CURRENCIES["USD"])
    return amount / info["rate"]

def format_currency(amount_usd, currency_code="USD", decimals=2):
    """Formats a USD amount into target currency string with appropriate symbol."""
    info = CURRENCIES.get(currency_code, CURRENCIES["USD"])
    converted = amount_usd * info["rate"]
    symbol = info["symbol"]
    if currency_code == "JPY":
        return f"{symbol}{converted:,.0f}"
    return f"{symbol}{converted:,.{decimals}f}"
