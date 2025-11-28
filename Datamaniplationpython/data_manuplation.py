import pandas as pd
import urllib.request
# df=pd.read_csv(r'data_sets\data.csv')
# # diffrent types of creating a data frame
# a=[1,2,3,4,5,6]
# df=pd.DataFrame(a)
# # makeing using dict
# a={'a':[1,2],'b':[2,3],'c':[3,4],'d':[4,6]}
# df=pd.DataFrame(a)
# # Acessinf element
# df['a']# using column
# df[['a','b']]#acessing multiple columns
# df.loc[0][0]#accesing alements like array 0 row 0 col
# df.at[0,'a']#row with index  and col with name
# #Adding new colum
# df['Name']=['jai'] 
# print(df['Name'].dtype)
# df.rename(columns={'a':'z'})# used to remame a col
# # df.fillna()using this we can fill na values with means or what you want
# df['new_value']=df['b'].apply(lambda x:x**2)# used apply a formula for particular colum
# # Data aggregating and groping
# df=pd.read_csv(r'data_sets\data.csv')
# grouped_mean=df.groupby('Product')['Value'].mean()
# Need to check group by workin

# Now read the HTML
url='https://en.wikipedia.org/wiki/List_of_highest_individual_scores_in_One_Day_International_cricket'
df = pd.read_html(url)
url = 'https://en.wikipedia.org/wiki/List_of_highest_individual_scores_in_One_Day_International_cricket'

req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0'}  # this is only a fake browser identity
)

html = urllib.request.urlopen(req).read()

tables = pd.read_html(html,match='Name')