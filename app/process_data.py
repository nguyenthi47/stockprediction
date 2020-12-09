import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import csv 

def save_tickers():
	resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	soup = bs.BeautifulSoup(resp.text)
	table = soup.find('table',{'class':'wikitable sortable'})
	tickers = []
	industries = []
	for row in table.findAll('tr')[1:]:
		ticker = row.findAll('td')[0].text[:-1]
		tickers.append(ticker)
		industry = row.findAll('td')[3].text[:-1]
		# print(industry)
		industries.append(industry)
	with open("tickers.pickle",'wb') as f:
		pickle.dump(tickers, f)
	with open("industries.csv",'w') as ff:
		wr = csv.writer(ff, delimiter = "\n")
		wr.writerow(industries)
	return tickers

def fetch_data():
	with open("tickers.pickle",'rb') as f:
		tickers=pickle.load(f)
	if not os.path.exists('stock_details'):
		os.makedirs('stock_details')
	start = dt.datetime(2010,1,1)
	today = date.today() 
	end = dt.datetime(today.year,today.month,today.day)
	ind = pd.read_csv('industries.csv')
	count = 0
	for ticker in tickers:
		print(ticker)
		try:
			df = web.DataReader(ticker, 'yahoo', start, end)
			df.to_csv('stock_details/{}.csv'.format(ticker))
		except:
			print("Error")
		continue

if __name__ == "__main__":
	save_tickers()
	fetch_data()