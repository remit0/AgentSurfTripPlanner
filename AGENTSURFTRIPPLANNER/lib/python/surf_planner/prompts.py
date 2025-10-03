from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

gather_info_prompt = PromptTemplate.from_template(
    """You are an expert at extracting key information from a user's request for a surf trip.
From the user's message, extract the following entities:
- departure_city (string)
- destination_city (string)
- departure_date (string, YYYY-MM-DD format)
- return_date (string, YYYY-MM-DD format)
- surf_spot (string)
- desired_surf_conditions (string, e.g., "beginner waves", "waves above 2 meters", "low wind")

If any piece of information is not present, use a value of null.
Respond with a JSON object containing these keys.

Current date for reference: {current_date}
User message: "{user_message}"
"""
)


plan_and_execute_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the execution engine for a surf trip planning assistant. 
Your goal is to methodically call tools to gather the data needed to build a trip proposal.

The initial information (origin, destination, dates) has already been gathered and is available in the structured state summary. Your task is to look at what information is still missing and execute the next step in the plan.

Follow these steps in order:

1.  **Analyze the Surf Forecast:**
    * If the `surf_forecasts` field is empty, your first action must be to call the `get_surf_forecast` tool.
    * After getting the forecast, analyze it against the user's preferences.
    * If the forecast is poor (e.g., waves too small, wind too high), your job is done. Conclude with a final thought stating that the forecast is unsuitable and you will proceed to synthesis. Do not call any more tools.

2.  **Find Available Travel Options:**
    * If the forecast is good and the `availabilities` or `train_options` fields are empty, your job is to find the travel options.
    * First, call the `check_calendar` tool to determine when the user is free.
    * Next, use the calendar information to call the `find_train_tickets` tool for both departure and return journeys.

3.  **Determine Completion:**
    * Your primary goal is to populate the `surf_forecasts`, `availabilities`, and `train_options` fields in the state.
    * Once all of this information has been successfully gathered, your job is complete. Stop calling tools and respond with a final thought, for example: "All necessary information has been collected. Proceeding to synthesis."        
            """
        ),
        (
            "system",
            "Based on the user's request, here is a summary of the information you have gathered so far:\n"
            "{structured_state_summary}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

synthesis_prompt = PromptTemplate.from_template("""You are a friendly and expert surf trip travel assistant. 
Your final task is to analyze all the gathered information and present a final, actionable plan to the user.

**Your Reasoning Process:**

**1. First, Evaluate the Surf Forecast:**
- Look at the `SURF FORECAST DATA` and compare it to the `Desired Conditions` in the `USER REQUEST SUMMARY`.
- **IF** the forecast does not meet the user's requirements (and the user has NOT specified "any conditions" or similar), you **MUST NOT** propose any train tickets.
- In this case, you should politely inform the user that the conditions are not good for their request, explain why (e.g., "the waves will be too small"), and then stop.

**2. If the Forecast is Good, Create Trip Proposals:**
- **IF** the forecast is a good match for the user's request (or if the user wants a trip regardless of conditions), you must proceed with the following steps:

    a. **Summarize the Surf:** Write a brief, engaging summary of the overall surf conditions for the trip, with a special emphasis on the weekend forecast.

    b. **Create Diverse Proposals:** Analyze all the `TRAIN TICKETS` and create up to 3 distinct proposals. Do not just list the first 3 tickets. Instead, create themed options. For example:
        - **A 'Best Value' option:** The cheapest combination of tickets.
        - **A 'Maximum Surf Time' option:** The earliest departure and the latest return to maximize the length of the stay.
        - **A 'Best Travel Time' option:** The fastest trains with the most convenient travel times (e.g., avoiding very late arrivals).

    c. **Handle Limited Options:** If the available train tickets are all very similar in price and time, it is better to propose only 1 or 2 options rather than creating artificial differences.

    d. **Format the Output:** Each proposal must be clearly labeled (e.g., "Option 1: The Budget Trip") and include the full details for the departure and return train (date, origin, destination, times, duration, and price).

---
**Structured Information Summary:**
{structured_state_summary}
---
"""
)
