from flask import request, Flask
from research_assistant import Agent, get_company_ticker, get_financial_statement, get_stock_price, get_company_headlines
from langchain_openai import ChatOpenAI

app = Flask(__name__)

@app.route('/ask_question', methods = ['POST', 'GET'])
def wealth_expert():
    data = request.get_json()
    user_query = data['user_query']
    print(user_query)
    model = ChatOpenAI(model="gpt-4o", openai_api_key = '**********')  
    tools = [get_company_ticker, get_financial_statement, get_stock_price, get_company_headlines]
    abot = Agent(model, tools)
    results = abot.graph.invoke({'user_input':user_query})
    return results

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
