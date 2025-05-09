'''
This module will contain the core logic for the NADO chatbot, 
interacting with the OpenAI Responses API.
'''

import openai
import os

# Load API key from environment variable
# Ensure OPENAI_API_KEY environment variable is set in your execution environment.
# For Streamlit deployment, this will typically be managed via Streamlit Secrets.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # In a real application, you might raise an error or handle this more gracefully.
    # For now, we'll print a warning and allow the script to proceed,
    # though API calls will fail.
    print("WARNING: OPENAI_API_KEY environment variable not set. API calls will fail.")
    # raise ValueError("OPENAI_API_KEY environment variable not set.")

SYSTEM_PROMPT = """You are NADO, the official success consultant for Health Provider Assist. Your role is to be a friendly, **strictly professional**, warm, real, and motivational pre-sales assistant. Your primary goal is to discover the user's situation regarding their NDIS provider journey (are they starting, stuck, or growing?) and then recommend the most suitable Health Provider Assist product (HPA NDIS Registration, HPA Portal + NADO, or HPA Plus). Guide users toward taking an action such as signing up for a package, booking a call/demo, or viewing package details on the website.

Key Instructions & Persona:
- Always maintain a **professional, direct, and informative tone**. Avoid casual language or conversational filler.
- **Do NOT use any emojis under any circumstances.**
- Write short, clear, and concise responses. Use bullet points and headings where helpful for clarity and directness.
- Your default opening question is: "Hi 👋 Are you looking to start your NDIS business, or do you already have one and want help to grow or stay compliant?"
- NEVER provide deep technical answers or documents.
- NEVER give away full solutions or detailed walkthroughs. Your purpose is to guide users to where they can get the full service from Health Provider Assist.
- Focus on conversion: encourage users to sign up, book a call, or explore packages.
- **Information Hierarchy:**
    - For any questions directly relating to product packages, their specific inclusions as detailed below, pricing, or discount codes, **you MUST prioritize the "Product Information" section provided below.** Do not use web search for these specific details unless the information is explicitly missing from the section below.
    - Use the `web_search_preview` tool to find information on the Health Provider Assist websites (healthproviderassist.com.au, hpaplus.com.au, portal.healthproviderassist.com.au) for general questions about services, company details, or topics NOT explicitly and fully covered in the "Product Information" section below.
    - If a web search is performed and the required information is not found or is unclear, clearly state that the specific information could not be found. Do not speculate or provide potentially incorrect details.
- Be factual and directly answer user questions based on the information hierarchy.

Product Information (Prioritize this for direct questions about packages, pricing):

**HPA NDIS Registration Packages (for new providers):**
*   **One-Time Registration – $3,500 (Incl. GST)**
    *   Perfect for: Starting small and staying in control.
    *   Includes: NDIS application, policies, audit prep, and 6 months of HPA PLUS.
    *   Benefits: Saves time, ensures compliance, gives control.
*   **Lifetime Registration – $10,000 (Currently 6,600 with code HPA25 - mention this discount!)**
    *   Perfect for: Peace of mind and long-term support.
    *   Includes: Everything in One-Time PLUS lifetime support for audits, updates, and questions.
*   **Business Package – $13,200 (or 2 x $7,250)**
    *   Perfect for: Growing fast with a brand.
    *   Includes: Everything in Lifetime PLUS custom branding, website, digital assets, social media, HPF access, and 12 months HPA PLUS.

**HPA PLUS — Business Operations & Daily Management Tool (for active providers managing clients, staff, operations):**
*   Key features: Roster staff & manage shifts, submit invoices & track payments, record case notes & incidents, store client files & staff records, HR tools for onboarding & compliance, document trail for audit prep, mobile + web access.
*   Pricing (after free trial/initial inclusion):
    *   1–5 users: $50/month
    *   6–40 users: $252/month or $2,400/year
    *   HPA clients discount: $126/month or $1,200/year

**HPA PORTAL — Training, Compliance, and Smart System Support (for compliance, training, business guidance):**
*   Key features: Full staff training library, upload & assign your own training, track completion & compliance, access policy guides & documents, 24/7 AI support with Nado (the more advanced AI, not you), push alerts for HR/training tasks, desktop-first experience.
*   **Nado (Built-in AI Assistant in HPA Portal):** Helps reduce errors and simplify compliance by answering complex questions and guiding task steps.
*   Pricing:
    *   Up to 25 staff: $299/month or $2,990/year.
    *   Enterprise (over 25 staff): Price by quote.

**Synergy:**
- Remind users: "Use HPA Plus to run your business. Use HPA Portal to stay compliant and train your team."

**Call to Actions (CTAs) & Links (Use these to guide users):**
*   **General How to Get Started:**
    1.  Visit: https://healthproviderassist.com.au
    2.  Click "HPA Packages"
    3.  Choose your package
    4.  Click "Sign Up"
    5.  Pick payment option
    6.  Enter details & submit.
    7.  Inform them: "After confirmation, reply or text your order number. You'll get your onboarding pack including: Service Agreement and NDIS Service List (we help you choose)."
*   **Motivational CTAs (use frequently, adapt to professional tone):**
    *   "Would you like assistance in choosing the best option?"
    *   "You can learn more at [insert relevant link here]. I can guide you if you provide the link or topic." (Adjusted for bot context)
    *   "Are you ready to explore your options now?"
    *   "Many find that starting sooner is beneficial. Shall we proceed?"
*   **Specific Links:**
    *   Health Provider Assist Main Site: https://healthproviderassist.com.au/
    *   Book a Consultation: https://healthproviderassist.com.au/book-online/
    *   HPA Packages Page: https://healthproviderassist.com.au/packages/
    *   HPA Portal Home: https://portal.healthproviderassist.com.au/
    *   HPA Portal Sign-Up: https://portal.healthproviderassist.com.au/membership-account/membership-levels/
    *   HPA Portal Android App: https://play.google.com/store/apps/details?id=com.au.hpaportal&hl=en
    *   HPA Portal Apple App: https://apps.apple.com/au/app/hpa-portal/id6743937739
    *   HPA Plus Site: https://hpaplus.com.au/
    *   HPA Plus Book a Demo: https://hpaplus.com.au/book-a-free-demo/
    *   HPA Plus Packages & Free Trial: https://hpaplus.com.au/packages/
    *   HPA Plus Android App: https://play.google.com/store/search?q=HPA%20PLUS&c=apps&hl=en
    *   HPA Plus Apple App: https://apps.apple.com/au/app/hpa-plus/id6470020643

Remember to adapt your response based on the user's stage (starting, stuck, or growing) to recommend the most relevant services.
"""

class Chatbot:
    def __init__(self):
        if OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None # API calls will not work
            print("Chatbot initialized without API key. API calls will be skipped.")
        
        self.current_response_id = None # For conversation continuity with Responses API
        # self.local_conversation_log = [] # Optional: for local debugging

    def get_opening_question(self) -> str:
        try:
            # Ensure the quotes here match exactly those in SYSTEM_PROMPT
            # The SYSTEM_PROMPT uses: "Hi 👋 ... compliant?"
            question_part = SYSTEM_PROMPT.split('Your default opening question is: "')[1]
            question = question_part.split('"')[0]
            return question
        except IndexError:
            # Fallback if parsing fails, maintaining a professional tone
            return "How can I assist you today with your NDIS provider journey?"

    def get_response(self, user_input: str) -> str:
        '''
        Gets a response from the OpenAI Responses API.
        Uses web_search_preview tool and server-side state management.
        '''
        if not self.client:
            return "Sorry, the chatbot is not configured with an API key. I cannot process your request."
        if not user_input:
            return "Please provide some input."

        try:
            # Structure for client.responses.create
            # Content for messages within the 'input' list should be a list of typed objects.
            messages_for_input_param = [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_input}]}
            ]

            api_params = {
                "model": "gpt-4o", 
                "input": messages_for_input_param,
                "tools": [{"type": "web_search_preview"}],
                "store": True # Enable server-side state management
            }

            if self.current_response_id:
                # For subsequent turns, provide the ID of the previous response
                api_params["previous_response_id"] = self.current_response_id
            
            # Ensure you are using a version of the openai library that supports .responses.create
            # This is the key change from .chat.completions.create
            response = self.client.responses.create(**api_params)

            assistant_reply = "Error: Could not extract assistant's reply." 
            
            # Expected structure for response.output (based on community examples for Responses API):
            # response.output is a list of message objects (typically one for the assistant).
            # Each message object has a 'role' and 'content' (a list of content parts).
            # Each content part has a 'type' (e.g., 'text') and the actual data (e.g., 'text' field).
            if response.output and isinstance(response.output, list) and len(response.output) > 0:
                assistant_message = response.output[0] # Assuming the first output is the assistant's
                if hasattr(assistant_message, 'content') and isinstance(assistant_message.content, list) and len(assistant_message.content) > 0:
                    for content_part in assistant_message.content:
                        if hasattr(content_part, 'type') and content_part.type == 'output_text' and hasattr(content_part, 'text'):
                            assistant_reply = content_part.text
                            break 
            
            # Update current_response_id for the next turn
            # Assuming the response object from .responses.create has an 'id' attribute for the response itself
            if hasattr(response, 'id'):
                self.current_response_id = response.id
            else:
                # Fallback or error if ID is not found as expected, as it's crucial for context
                print("Warning: Could not retrieve 'id' from response object for state continuity.")
                self.current_response_id = None # Reset to avoid sending an invalid ID

            return assistant_reply

        except AttributeError as e:
            # This might happen if self.client.responses.create does not exist (e.g. older SDK version)
            print(f"AttributeError: {e}. Does your OpenAI Python SDK version support 'client.responses.create'?")
            return "Sorry, there was a configuration issue with the AI service (AttributeError)."
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}. Status Code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")
            error_body = getattr(e, 'body', None)
            error_message_detail = "No further details."
            if error_body and 'error' in error_body and isinstance(error_body['error'], dict) and 'message' in error_body['error']:
                error_message_detail = error_body['error']['message']
            elif isinstance(error_body, dict) and 'message' in error_body: # some errors have message at top level of body
                 error_message_detail = error_body['message']

            if hasattr(e, 'status_code') and e.status_code == 401: # Unauthorized
                 return f"Sorry, there was an authentication issue with the AI service: {error_message_detail}"
            if hasattr(e, 'status_code') and e.status_code == 400:
                 # More detailed error for bad request if possible
                 return f"Sorry, the AI service reported a problem with the request: {error_message_detail}"
            if hasattr(e, 'status_code') and e.status_code == 429:
                 return f"Sorry, the AI service rate limit has been exceeded: {error_message_detail}"
            return f"Sorry, I encountered an API error ({type(e).__name__}): {error_message_detail}. Please try again later."
        except Exception as e:
            print(f"An unexpected error occurred: {type(e).__name__} - {e}")
            return "Sorry, I encountered an unexpected error. Please try again."

# Example Usage (for local testing of the core logic later)
if __name__ == "__main__":
    print("Initializing Chatbot...")
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set. The chatbot will run with placeholder responses.")
    
    bot = Chatbot()
    
    print("\n--- Simulating Conversation ---")
    # Test with the default opening question to see how it might be handled or if the bot uses it.
    # In a real scenario, the Streamlit app would send the first user message.
    # user_query1 = SYSTEM_PROMPT.split('Your default opening question is: "')[1].split('"')[0]
    user_query1 = "Hi, I'm thinking about starting an NDIS business."
    print(f"User > {user_query1}")
    bot_reply1 = bot.get_response(user_query1)
    print(f"NADO > {bot_reply1}")

    if OPENAI_API_KEY and bot_reply1 and "Sorry" not in bot_reply1 and "Error" not in bot_reply1:
        user_query2 = "Tell me about HPA Plus and its pricing."
        print(f"User > {user_query2}")
        bot_reply2 = bot.get_response(user_query2)
        print(f"NADO > {bot_reply2}")

        if bot_reply2 and "Sorry" not in bot_reply2 and "Error" not in bot_reply2:
            user_query3 = "What about the Lifetime Registration package? Is there a discount code?"
            print(f"User > {user_query3}")
            bot_reply3 = bot.get_response(user_query3)
            print(f"NADO > {bot_reply3}")
    else:
        print("\nSkipping further conversation simulation due to previous error or missing API key.")

    print("\n--- Simulation Complete ---") 