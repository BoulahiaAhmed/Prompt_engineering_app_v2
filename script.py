import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
import typing_extensions as typing
import logging
import streamlit as st
import json
import os
from dotenv import load_dotenv

# Define TypedDict for Gemini response
class GeminiResponse(typing.TypedDict):
    rule_name: str
    label: bool
    part: list[str]
    suggestion: list[str]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
#load_dotenv()
#GOOGLE_API_KEY = os.getenv('API_KEY')
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure Google Generative AI with API Key
genai.configure(api_key=GOOGLE_API_KEY)

# Define Generation and Safety Configurations
genai_generation_config = GenerationConfig(
    candidate_count=1,
    max_output_tokens=800,
    temperature=0,
    response_mime_type="application/json",
    response_schema=GeminiResponse
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

def gemini_answer(prompt: str, model: genai.GenerativeModel) -> typing.Optional[str]:
    """Generate content using the Gemini model and return the response text."""
    try:
        response = model.generate_content(prompt, generation_config=genai_generation_config, safety_settings=safety_settings)
        response_text = response.parts[0].text
        logger.info(f"Response: {response_text}")
        return response_text
    except json.JSONDecodeError:
        logger.error("Invalid JSON output string")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def create_rules_list(input_string: str):
    seperator = '##'
    return input_string.split(seperator)


def inference(system_message: str, model_name: str, rules_list: list[str], sales_deck: str) -> typing.Optional[str]:
    """Perform inference using the Gemini model and return the generated response."""
    output_list=[]

    for rule in rules_list:
        input_text = f"""
        The rule is: {rule}
        The sales deck to evaluate is: {sales_deck}
        """ 

        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_message)
        model_output = gemini_answer(input_text, model)
        output_list.append(model_output)

    return output_list


def main():
    """Main function to run the script."""
    sales_deck_example= """
    Welcome to BrightFuture Investments! We are dedicated to providing top-notch investment opportunities tailored to your financial goals. With our expert team and innovative strategies, your financial future is in safe hands. At BrightFuture Investments, we understand the complexities of the financial market and strive to simplify the investment process for you. Our mission is to help you achieve your financial aspirations with confidence and ease.
    BrightFuture Investments leverages cutting-edge algorithms and market insights to maximize your returns. Our team of experts has developed a sophisticated investment strategy that has historically delivered exceptional results. Many of our clients have seen their investments grow significantly, often doubling within a short period. While we always emphasize that past performance does not guarantee future results, our track record speaks volumes about our capability and dedication. Our focus on minimizing risk while maximizing returns sets us apart in the industry. Our platform consistently outperforms the competition, making it the preferred choice for savvy investors. We pride ourselves on our ability to deliver superior returns and unparalleled service. Many of our clients achieve their financial independence much faster than they anticipated, thanks to our innovative approach. By choosing BrightFuture Investments, you are aligning yourself with a team that prioritizes your financial success and is committed to helping you reach your goals.
    At BrightFuture Investments, we offer personalized investment plans tailored to your unique needs and objectives. Our comprehensive approach ensures that every aspect of your financial journey is carefully considered and optimized for maximum growth. From the initial consultation to ongoing portfolio management, we are with you every step of the way, providing expert guidance and support.
    Our advanced technology and analytical tools enable us to stay ahead of market trends and make informed investment decisions. This proactive approach allows us to capitalize on opportunities and mitigate risks effectively. Our clients benefit from our deep market knowledge and strategic insights, which are integral to achieving consistent and impressive returns.
    Moreover, we are committed to transparency and integrity in all our dealings. Our clients have access to detailed reports and updates on their investment performance, ensuring they are always informed and confident in their financial decisions. We believe in building long-term relationships based on trust and mutual success.
    In summary, BrightFuture Investments is your partner in achieving financial success. With our proven strategies, expert team, and commitment to excellence, you can rest assured that your investments are in capable hands. Join us today and take the first step towards a brighter financial future. Let us help you turn your financial dreams into reality with confidence and peace of mind.
    """

    result = inference("gemini-1.5-flash", ["Inclusion of Risk Warnings"], sales_deck_example)
    if result:
        print(result)

if __name__ == "__main__":
    main()
