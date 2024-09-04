from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from typing import TypedDict, Annotated, List
from langchain_core.pydantic_v1 import BaseModel
import operator
from langgraph.graph import StateGraph, END
import feedparser 
from bs4 import BeautifulSoup
import requests
import yfinance as yf
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
import flask


from langchain.tools import DuckDuckGoSearchRun
import re
from langchain_openai import ChatOpenAI
import os


search = DuckDuckGoSearchRun()

@tool
def get_company_ticker(company_name : str):
    """get company's NSI ticker"""    
    n = search.run(f'{company_name} ticker of NSI')
    return re.search(r'\(.*?:NSI\)', n).group()


@tool
def get_financial_statement(ticker : str):
    """ get financial statement of the company, using company's ticker"""
    
    if "." in ticker:
        ticker=ticker.split(".")[0]
    else:
        ticker=ticker
    ticker=ticker+".NS"    
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1]>=3:
        balance_sheet=balance_sheet.iloc[:,:3]    # Remove 4th years data
    balance_sheet=balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

@tool
def get_stock_price(ticker : str):
    """ get stock price of the company, using company's ticker"""
    
    history=5
    if "." in ticker:
        ticker=ticker.split(".")[0]
    ticker=ticker+".NS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df=df[["Close","Volume"]]
    df.index=[str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date",inplace=True)
    df=df[-history:]
    return df.to_string()

@tool
def get_company_headlines(company_name):
    """ get news headlines of the company, using company's name"""
                
    company_name = '+'.join(company_name.split(' '))
    google_news_url = f'https://news.google.com/rss/search?q={company_name}+ticker+stock+news+india&hl=en-US&gl=US&ceid=US:en'
    feed = feedparser.parse(google_news_url)

    news_entries = []
    for entry in feed['entries'][:6]:
        news_entries.append(entry['published'] +' '+ entry['title'])
    
    return news_entries


class Headlines(BaseModel):
    headlines: List[str]


class AgentState(TypedDict):

    company_ticker_messages: Annotated[list[AnyMessage], operator.add]
    stock_information_messages: Annotated[list[AnyMessage], operator.add]
    stock_headlines_messages: Annotated[list[AnyMessage], operator.add]
    stock_news_messages: Annotated[list[AnyMessage], operator.add]
    stock_analyser_messages: Annotated[list[AnyMessage], operator.add]

    user_input: str
    ticker :str
    content: str
    headlines: List[str]
    response: str


class Agent:

    def __init__(self, model, tools):

        graph = StateGraph(AgentState)

        graph.add_node("extract_company_ticker", self.extract_company_ticker)
        graph.add_node("get_stock_information", self.get_stock_information)
        graph.add_node("get_stock_headlines", self.get_stock_headlines)
        graph.add_node("get_stock_news", self.get_stock_news)
        graph.add_node("stock_analyser", self.stock_analyser)


        graph.add_node("take_action_to_find_ticker", self.take_action_to_find_ticker)
        graph.add_node("take_action_to_find_stock_information", self.take_action_to_find_stock_information)
        graph.add_node("take_action_to_find_stock_headline", self.take_action_to_find_stock_headline)
        

        graph.add_conditional_edges(
            "extract_company_ticker",
            self.action_exist_for_company_ticker,
            {True: "take_action_to_find_ticker", False: "get_stock_information"}
        )

        graph.add_edge("take_action_to_find_ticker", "extract_company_ticker")

        graph.add_conditional_edges(
            "get_stock_information",
            self.action_exist_for_stock_information,
            {True: "take_action_to_find_stock_information", False: "get_stock_headlines"}
        )

        graph.add_edge("take_action_to_find_stock_information", "get_stock_information")


        graph.add_conditional_edges(
            "get_stock_headlines",
            self.action_exist_for_stock_headline,
            {True: "take_action_to_find_stock_headline", False: "get_stock_news"}
        )

        graph.add_edge("take_action_to_find_stock_headline", "get_stock_headlines")

        graph.add_edge("get_stock_news", "stock_analyser")

        graph.add_edge("stock_analyser", END)
        
        graph.set_entry_point("extract_company_ticker")
        
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.search = DuckDuckGoSearchRun()

    def extract_company_ticker(self, state: AgentState):

        TICKER_IDENTIFIER_PROMPT = '''
        From the user query, first identify the company name. Then pass the company name to get_company_ticker function.
        From the response of function, identify the company ticker, and only return the ticker. You should not answer the user query.
        For example, if the company identified is State Bank of India, the final expected output should be "SBIN"
        '''
        
        messages = state.get('company_ticker_messages')
        
        messages = [SystemMessage(content=TICKER_IDENTIFIER_PROMPT), HumanMessage(content=state['user_input'])] + messages
        response = self.model.invoke(messages)

        return {'company_ticker_messages': [response]}
    
    def get_stock_information(self, state: AgentState):

        GET_STOCK_INFORMATION_PROMPT = '''
        For the given ticker, use get_stock_price function to get the stock price and 
        then use the get_financial_statement function to get the financial statement and return the result after combining both'''

        messages = state.get('stock_information_messages')
        
        ticker = state.get('company_ticker_messages')[-1].content

        messages = [SystemMessage(content=GET_STOCK_INFORMATION_PROMPT), HumanMessage(content=ticker)] + messages
        reponse = self.model.invoke(messages) 

        return {'stock_information_messages': [reponse]}
    
    def get_stock_headlines(self, state: AgentState):

        GET_STOCK_HEADLINE_PROMPT = '''From the given user query, identify the company name and use get_company_headlines function to get the latest company news.
        Then, from all the headlines, identify the most relevant headlines which are related to company's stocks 
        (like increase or decrease in profit) or other news which depict the growth or decline of the company 
        (it could be about their new projects, awards, new deals won or loss of deal).
        Neglect the headlines which is ambiguous. Finally pick 4 maximum Headlines.
        '''
        messages = state.get('stock_headlines_messages')
        
        messages = [SystemMessage(content=GET_STOCK_HEADLINE_PROMPT), HumanMessage(content=state['user_input'])] + messages
        
        headlines_structured = self.model.with_structured_output(Headlines).invoke(messages)
        headlines_raw = self.model.invoke(messages)
        

        return {'headlines': headlines_structured, 'stock_headlines_messages': [headlines_raw]}

    def get_stock_news(self, state: AgentState):

        headlines = state.get('headlines') or []
        existing_content = '\nHere are some latest headlines about the stock\n\n'

        for headline in headlines.headlines:
            print(headline) ## added for debugging purpose
        
        return {'content': existing_content}
    
    def stock_analyser(self, state: AgentState):

        STOCK_ANALYSER_PROMPT = '''You are an stock market expert. Based on the Financial Data and NEWS about the company below, 
        answer the user query. Utilize all the information below as needed. Make sure to do thorough analysis before answering: 
        {content}'''

        content = state.get('content') or ''

        stock_information = 'Here is the Stock information of the company = \n' + state.get('stock_information_messages')[-1].content +'\n'+content
        
        print('&&&&&&&&&&&&&&&&&',stock_information,'&&&&&&&&&&&&&') ## added for debugging purpose
        
        response = self.model.invoke([SystemMessage(content=STOCK_ANALYSER_PROMPT.format(content = stock_information)),
        HumanMessage(content=state['user_input'])])
        print(response)
    
        return {"response": [response]}
    

    def take_action_to_find_ticker(self, state: AgentState):
        tool_calls = state['company_ticker_messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the Model")
        return {'company_ticker_messages': results}
    
    def take_action_to_find_stock_information(self, state: AgentState):
        tool_calls = state['stock_information_messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the Model!")
        return {'stock_information_messages': results}
    

    def take_action_to_find_stock_headline(self, state: AgentState):
        tool_calls = state['stock_headlines_messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the Model!")
        return {'stock_headlines_messages': results}
    
    def action_exist_for_company_ticker(self, state: AgentState):
        result = state['company_ticker_messages'][-1]
        return len(result.tool_calls) > 0
    def action_exist_for_stock_information(self, state: AgentState):
        result = state['stock_information_messages'][-1]
        return len(result.tool_calls) > 0
    
    def action_exist_for_stock_headline(self, state: AgentState):
        result = state['stock_headlines_messages'][-1]
        return len(result.tool_calls) > 0
    

if __name__ == '__main__':

    model = ChatOpenAI(model="gpt-4o")  

    tools = [get_company_ticker, get_financial_statement, get_stock_price, get_company_headlines]

    abot = Agent(model, tools)

    user_query = 'Should I invest in Yes bank?'
    results = abot.graph.invoke({'user_input':user_query})
