import os
from dotenv import load_dotenv
from groq import Groq
import typing_extensions as typing
import logging
import json
import streamlit as st
from typing import List, Optional

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model names
llama3_1='llama-3.1-70b-versatile'
mixtral='mixtral-8x7b-32768'
gemma='gemma2-9b-it'


# Data model for LLM to generate
class Desired_output(BaseModel):
    rule_name: bool
    label: str
    part: list[str]
    suggestion: list[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def groq_model_generation(prompt: str, system_message: str, model: str) -> dict:
    """Model names: llama3_1, mixtral, gemma"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{system_message}"
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content
        logger.info(f"Response: {result}")

        # Parse result and raise exception if it's not valid JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error("Invalid JSON output string")
            raise

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def groq_inference(system_message: str, model_name: str, rules_list: list[str], sales_deck: str) -> typing.Optional[str]:
    """Perform inference using the groq api models and return the generated response."""
    output_list = []
    for rule in rules_list:
        input_text = f"""
        The rule is: {rule}
        The sales deck to evaluate is: {sales_deck}
        Your MUST provide an output in JSON representation with the following fields:
        "rule_name",
        "label",
        "part",
        "suggestion"
        """
        model_output = groq_model_generation(input_text, system_message, model_name)
        output_list.append(model_output)

    return output_list


def video_card_generation(transcript: str, model: str) -> str:
    """Model names: llama3_1, mixtral, gemma"""
    system_message = """
    Your task is to generate a concise summary from the given video transcript.
    Please follow these instructions return a markdown text:

    1. Extract Key Information:
    - Identify the company name.
    - Determine the industry, if applicable.
    - Summarize the product or service being discussed.

    2. Output Format:
    - **Company Name**: [Extracted company name]
    - **Industry**: [Extracted industry, if available]
    - **Product Summary**: [Brief summary of the product or service]

    Example:
    For a video discussing "FinGuard’s new portfolio management tool designed to help investors track and optimize their asset allocations," your output might look like:

    - Company Name: FinGuard
    - Industry: Financial Services
    - Product Summary: A portfolio management tool that assists investors in tracking and optimizing their asset allocations for improved investment outcomes.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{system_message}"
                },
                {
                    "role": "user",
                    "content": f"Here is the transcript to use: {transcript}",
                }
            ],
            model=model,
            temperature=0,
        )

        result = response.choices[0].message.content
        logger.info(f"Response: {result}")
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


# Local testing
default_sales_deck="""Welcome to BrightFuture Investments! We are dedicated to providing top-notch investment opportunities tailored to your financial goals. With our expert team and innovative strategies, your financial future is in safe hands. At BrightFuture Investments, we understand the complexities of the financial market and strive to simplify the investment process for you. Our mission is to help you achieve your financial aspirations with confidence and ease.
    BrightFuture Investments leverages cutting-edge algorithms and market insights to maximize your returns. Our team of experts has developed a sophisticated investment strategy that has historically delivered exceptional results. Many of our clients have seen their investments grow significantly, often doubling within a short period. While we always emphasize that past performance does not guarantee future results, 
    our track record speaks volumes about our capability and dedication. Our focus on minimizing risk while maximizing returns sets us apart in the industry. Our platform consistently outperforms the competition, making it the preferred choice for savvy investors. We pride ourselves on our ability to deliver superior returns and unparalleled service. Many of our clients achieve their financial independence much faster than they anticipated, thanks to our innovative approach. 
    By choosing BrightFuture Investments, you are aligning yourself with a team that prioritizes your financial success and is committed to helping you reach your goals.
    At BrightFuture Investments, we offer personalized investment plans tailored to your unique needs and objectives. Our comprehensive approach ensures that every aspect of your financial journey is carefully considered and optimized for maximum growth. From the initial consultation to ongoing portfolio management, we are with you every step of the way, providing expert guidance and support.
    Our advanced technology and analytical tools enable us to stay ahead of market trends and make informed investment decisions. This proactive approach allows us to capitalize on opportunities and mitigate risks effectively. Our clients benefit from our deep market knowledge and strategic insights, which are integral to achieving consistent and impressive returns.
    Moreover, we are committed to transparency and integrity in all our dealings. Our clients have access to detailed reports and updates on their investment performance, ensuring they are always informed and confident in their financial decisions. We believe in building long-term relationships based on trust and mutual success.
    In summary, BrightFuture Investments is your partner in achieving financial success. With our proven strategies, expert team, and commitment to excellence, you can rest assured that your investments are in capable hands. Join us today and take the first step towards a brighter financial future. Let us help you turn your financial dreams into reality with confidence and peace of mind.
    """

default_system_message="""You are a compliance officer.
    Your task is to understand the following rule and verify its adherence in the given sales deck.

    The steps are as follows:
    Understand the given rule.
    Augment the rule with additional vocabulary related to financial products.
    Evaluate the following sales deck: to determine if it respects the rule.

    Provide the output in JSON format with the following fields:
    rule_name (str): The name of the rule being applied.
    label (bool): return true if the rule is respected else return false.
    part (list[str]): Specific sections or aspects of the sales deck evaluated, including relevant details.
    suggestion (list[str]): Recommended changes or improvements to ensure compliance with the rule.

    Example JSON output structure:
    {
    "rule_name",
    "label",
    "part",
    "suggestion"
    }
    """

rules_list=["fairness", "grammatically correct"]

res = groq_inference(default_system_message, llama3_1, rules_list, default_sales_deck)
print(res)
