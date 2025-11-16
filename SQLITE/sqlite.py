import sqlite3
connection=sqlite3.connect('Example.db')
cursor=connection.cursor()
# cursor.execute('create Table IF Not Exists employees(id Integer Primary key,name Text Not Null,age Integer,department text)')
# connection.commit()
# cursor.execute('select *from employees')
# cursor.execute('''Insert Into employees(name,age,department) values('jai','24','datascience')''')
# cursor.execute('''Insert Into employees(name,age,department) values('Bharath','26','datascience')''')
# cursor.execute('''Insert Into employees(name,age,department) values('Sai','22','datascience')''')
# connection.commit()
# Quary the data from SQL
# cursor.execute('Select *from employees')
cursor.execute('Select *from sale')
rows=cursor.fetchall()
for row in rows:
    print(row)
cursor.execute(''' delete from employees where name='Sai' ''')
cursor.execute('create Table IF Not Exists sale(id Integer Primary key,name Text Not Null,age Integer,department text)')
sale_list=[('person1','22','grosery'),('person2','44','Washing utazils')]
cursor.executemany('''  INSERT INTO sale(name,age,department) values(?,?,?) ''',sale_list )
connection.commit()
# update data in the tablec
# cursor.execute(''' UPDATE employees Set age=22 where name='jai' ''' )
# connection.commit()
connection.close()