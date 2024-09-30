import streamlit as st
from script import gemini_answer, create_rules_list, inference
from groq_models import groq_inference, video_card_generation
from video_processing import transcribe_audio_with_whisper, extract_audio_from_video
import time
import os

default_sales_deck="""Welcome to BrightFuture Investments! We are dedicated to providing top-notch investment opportunities tailored to your financial goals. With our expert team and innovative strategies, your financial future is in safe hands. At BrightFuture Investments, we understand the complexities of the financial market and strive to simplify the investment process for you. Our mission is to help you achieve your financial aspirations with confidence and ease.
    BrightFuture Investments leverages cutting-edge algorithms and market insights to maximize your returns. Our team of experts has developed a sophisticated investment strategy that has historically delivered exceptional results. Many of our clients have seen their investments grow significantly, often doubling within a short period. While we always emphasize that past performance does not guarantee future results, our track record speaks volumes about our capability and dedication. Our focus on minimizing risk while maximizing returns sets us apart in the industry. Our platform consistently outperforms the competition, making it the preferred choice for savvy investors. We pride ourselves on our ability to deliver superior returns and unparalleled service. Many of our clients achieve their financial independence much faster than they anticipated, thanks to our innovative approach. By choosing BrightFuture Investments, you are aligning yourself with a team that prioritizes your financial success and is committed to helping you reach your goals.
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

default_rules = """fairness##grammatically correct"""


# Define the main function
def main():
    # Set the title of the app
    st.title('Poc: Prompt Testing & Enhancement')
    st.divider()
    st.subheader("🎬 Video Upload, Audio Extraction, and Transcription")

    # File uploader for video files
    video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])

    if video_file is not None:
        # Create the directory if it doesn't exist
        temp_video_dir = "temp_video"
        os.makedirs(temp_video_dir, exist_ok=True)
        # Save the uploaded video to the directory
        temp_video_path = os.path.join(temp_video_dir, video_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        
        if video_file is not None:
            # Create the directory for the audio if it doesn't exist
            temp_audio_dir = "temp_audio"
            os.makedirs(temp_audio_dir, exist_ok=True)
            # Define the path for the extracted audio file
            temp_audio_path = os.path.join(temp_audio_dir, "extracted_audio.mp3")
            # Extract audio from the video
            audio_path = extract_audio_from_video(temp_video_path, temp_audio_path)
        
        st.success("Audio extracted successfully!")

        # Display the video
        st.video(video_file)

        # Transcribe the audio using Whisper
        st.write("Transcribing audio...")
        sales_deck = transcribe_audio_with_whisper(audio_path)
        st.text_area("Video Transcript:", sales_deck, height=250)
    st.divider()
    st.subheader('✨ AI Model Selection')
    # Dropdown to select the model
    #model_name = st.selectbox("Select Model", ['gemini-1.5-flash', 'gemini-1.5-pro-latest'])
    appearing_model_name = st.radio("Select Model", ['gemini-1.5-flash', 'gemini-1.5-pro-latest', 'llama-3.1-70b', 'llama-3.2-90b', 'mixtral-8x7b', 'gemma2-9b'], horizontal=True)
    if appearing_model_name == 'gemini-1.5-flash':
        model_name = 'gemini-1.5-flash'
        st.info("Rate limit: 15 Request Per Minute")

    if appearing_model_name == 'gemini-1.5-pro-latest':
        model_name = 'gemini-1.5-pro-latest'
        st.info("Rate limit: 2 Request Per Minute")

    if appearing_model_name == 'llama-3.1-70b':
        model_name = 'llama-3.1-70b-versatile'
        st.info("Rate limit: 30 Request Per Minute")

    if appearing_model_name == 'llama-3.2-90b':
        model_name = 'llama-3.2-90b-text-preview'
        st.info("Rate limit: 30 Request Per Minute")

    if appearing_model_name == 'mixtral-8x7b':
        model_name = 'mixtral-8x7b-32768'
        st.info("Rate limit: 30 Request Per Minute")

    if appearing_model_name == 'gemma2-9b':
        model_name = 'gemma2-9b-it'
        st.info("Rate limit: 30 Request Per Minute")

    # st.divider()
    # st.subheader('Enter Sales Deck to evaluate here: ')
    # sales_deck = st.text_area("Sales Deck:", value=default_sales_deck, height=250)

    # Input for rules
    st.divider()
    st.subheader('👮 Enter rules here')
    st.write("\nIf you want to enter more than 1 rule use ## as a separator, \nexample:\n\n Fair, Clear, and Not Misleading ## Inclusion of Risk Warnings ")
    rules_string = st.text_area("Rules:", value=default_rules, height=200)
    rules_list = create_rules_list(rules_string)

    st.divider()
    st.subheader('🧩 Prompt Engineering')
    st.write("Modify this prompt to evaluate the model output")
    # Input for prompt
    system_message = st.text_area("Prompt:", value=default_system_message, height=500)

    st.divider()
    st.subheader('Model Output')
    # Call the generate function
    generate_output = st.button('Generate output')
    if generate_output:
        start = time.time()
        with st.spinner(text="Generation In progress..."):
            if 'gemini' in model_name:
                output = inference(system_message, model_name, rules_list, sales_deck)
            else:
                output = groq_inference(system_message, model_name, rules_list, sales_deck)
        end = time.time()
        st.write(f"Generation Duration: {end-start:.2f} seconds")
        for elm in output:
            st.json(elm)

    st.divider()
    st.subheader('Product card')
    generate_model_card = st.button('Product card')
    if generate_model_card:
        with st.spinner(text="Generation In progress..."):
            video_card = video_card_generation(sales_deck, model_name)
        st.markdown(video_card)



# Run the app
if __name__ == "__main__":
    main()
