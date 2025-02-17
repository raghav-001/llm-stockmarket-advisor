import requests
from langchain.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

# Initialize the LLM
model = Ollama(
    model="llama3:latest",
)

# Define a function to fetch stock data from an external API
def get_stock_data(stock_name: str):
    api_url = f"https://finnhub.io/api/v1/quote?symbol={stock_name}&token=API_KEY"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()  # Assuming the response is in JSON format
    else:
        return {"error": "Failed to fetch stock data"}

@cl.on_chat_start
async def on_chat_start():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a stock market advisor using real-time data to help users decide whether to buy or hold stocks.
                
                
                Using this information provided, provide a recommendation on whether the stock is worth buying, holding, or selling.
                Consider the current price, price change, and other indicators in your recommendation.
                """,
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    # Automatically ask the user to input a stock symbol as soon as the server is started
    msg = cl.Message(content="Please enter the stock symbol you'd like to analyze.")
    await msg.send()

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    stock_name = message.content.strip()  # Get stock name from user input
    stock_data = get_stock_data(stock_name)  # Fetch stock data using the API

    if "error" in stock_data:
        recommendation = "Sorry, I couldn't retrieve stock data at the moment."
    else:
        # Include the stock data in the prompt template for analysis
        stock_info = {
            "current_price": stock_data["c"],
            "price_change": stock_data["d"],
            "percent_change": stock_data["dp"],
            "high_price": stock_data["h"],
            "low_price": stock_data["l"],
            "open_price": stock_data["o"],
            "previous_close": stock_data["pc"],
            "timestamp": stock_data["t"]
        }

        # Pass the stock data into the prompt for decision-making
        prompt_input = {
            "question": f"Based on the stock data for {stock_name}:\n{stock_info}\nWhat is your recommendation on whether to buy, hold, or sell the stock?"
        }

        msg = cl.Message(content="")

        # Stream the recommendation and then ask for another stock symbol
        for chunk in await cl.make_async(runnable.stream)(
            prompt_input,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
        ):
            await msg.stream_token(chunk)
        
        # Add a message asking for another stock symbol after streaming the recommendation
        msg.content += "\n\n\nPlease enter another stock symbol you'd like to analyze."
        
        # Send the recommendation to the user
        await msg.send()
