"""
Transcript templates with variable placeholders
"""

# Original transcript with exact timing and natural pauses
TRANSCRIPT_TEMPLATE = """Hi, {customer_name}. I'm Kshitij, your dedicated travel advisor from Thirty Sundays. I'll be helping you plan your {destination} trip. I just wanted to put a face to the name so that you know who you're speaking with. At Thirty Sundays, we specialize in planning couple holidays. Every hotel, every activity is curated from a couple's land. So, uh, {destination} could be a great place for couples, {customer_name}, uh, but you need to know where to go, uh, and that is where we come in. I'll be calling you soon, {customer_name}, to help you plan that perfect {destination} holiday."""

# Default variable values (from original video)
DEFAULT_VARIABLES = {
    "customer_name": "Anuj Ji",
    "destination": "Bali"
}

# Variable occurrences in the template (for reference)
VARIABLE_OCCURRENCES = {
    "customer_name": 3,  # Hi, {customer_name} | couples, {customer_name} | soon, {customer_name}
    "destination": 3     # {destination} trip | {destination} could be | perfect {destination} holiday
}

# Template validation patterns
REQUIRED_VARIABLES = ["customer_name", "destination"]